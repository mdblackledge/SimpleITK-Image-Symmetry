import SimpleITK as sitk
import numpy as np
from scipy.optimize import minimize
from pywt import wavedecn

class ImageSymmetry(object):

    def plane_normalised(self, plane):
        if not type(plane) is np.ndarray:
            plane = np.array(plane)
        if not np.abs(np.sum(plane[0:-1]**2) - 1.0) < 1e-10:
            return False
        return True

    def normalise_plane(self, plane):
        norm = np.sqrt(np.sum(np.array(plane[0:-1])**2))
        return np.array(plane)/norm

    def cartesian_2_polar(self, plane):
        if not self.plane_normalised(plane):
            raise Exception("Input plane contents not normalised")
        plane_polar = np.zeros(len(plane)-1)
        plane_polar[-1] = plane[-1]
        plane_polar[-2] = np.arcsin(plane[-2])
        for i in range(len(plane_polar)-3, -1, -1):
            plane_polar[i] = np.arcsin(plane[i+1] / np.prod(np.cos(plane_polar[i+1:-1])))
        return plane_polar

    def polar_2_cartesian(self, plane_polar):
        plane = np.zeros(len(plane_polar)+1)
        plane[0] = np.prod(np.cos(plane_polar[0:-1]))
        for i in range(1, len(plane)-2):
            plane[i] = np.sin(plane_polar[i-1]) * np.prod(np.cos(plane_polar[i:-1]))
        plane[-2] = np.sin(plane_polar[-2])
        plane[-1] = plane_polar[-1]
        return  plane

    def __reflection_cost__(self, plane_polar, im):
        plane = self.polar_2_cartesian(plane_polar)
        imN = self.reflect_image(plane, im)
        cost = np.mean(np.abs(sitk.GetArrayFromImage(im-imN)))
        return cost

    def reflect_image(self, plane, im):
        trans = self.reflection_transform(plane)
        imN = sitk.Resample(im, im, trans, sitk.sitkLinear, 0.0, im.GetPixelID())
        return imN

    def plane_of_reflection(self, im, plane=None, levels=(2, 0)):
        if plane is None:
            plane = np.zeros(len(im.GetSize())+1)
            plane[0] = 1.0
        if not self.plane_normalised(plane):
            raise Exception("Input plane is not normalised")
        origin = im.GetOrigin()
        shape = np.array(im.GetSize())
        spacing = np.array(im.GetSpacing())
        plane_polar = self.cartesian_2_polar(plane)
        for level in levels:
            arr = wavedecn(sitk.GetArrayFromImage(im), 'db1', level=level)[0]
            im_ = sitk.GetImageFromArray(arr)
            im_.SetSpacing(shape / arr.shape[::-1] * spacing)
            im_.SetOrigin(origin + 0.5 * (im_.GetSpacing() - spacing))
            plane_polar = minimize(self.__reflection_cost__, plane_polar, (im_), method='Nelder-Mead', tol=1e-10).x
        plane = self.polar_2_cartesian(plane_polar)
        return plane

    def reflection_matrix(self, plane):
        mat = np.zeros((len(plane), len(plane)))
        for i in range(len(plane)-1):
            for j in range(len(plane)):
                if i == j:
                    mat[i, j] = 1 - 2 * plane[i] * plane[j]
                else:
                    mat[i, j] = - 2 * plane[i] * plane[j]
        mat[-1, -1] = 1.0
        return mat

    def reflection_transform(self, plane):
        trans_arr = self.reflection_matrix(plane)
        trans = sitk.AffineTransform(len(plane)-1)
        trans_params = []
        for i in range(len(plane)-1):
            trans_params = np.r_[trans_params, trans_arr[i, 0:-1].ravel()]
        trans_params = np.r_[trans_params, trans_arr[0:-1, -1].ravel()]
        trans.SetParameters(trans_params)
        return trans

    def plane_2d(self, x, plane):
        a = plane[0]
        b = plane[1]
        c = plane[2]
        return (a * x + c) / (-1. * b)

    def plane(self, X, plane):
        d = plane[-1]
        plane = plane[0:-1]
        return (np.einsum("ij,j->i", X, plane[0:-2]) + d)/(-1.*plane[-1])


if __name__ == "__main__":

        from scipy.misc import face
        import matplotlib.pyplot as pl

        image_sym = ImageSymmetry()

        # Create a mock image with symmetry
        arr = face(gray=True).astype('float')
        arr = np.pad(arr, ((arr.shape[0], arr.shape[0]), (arr.shape[1], arr.shape[1])), 'constant', constant_values=0.0)

        im = sitk.GetImageFromArray(arr)
        im.SetOrigin((-arr.shape[1]/2, -arr.shape[0]/2))
        plane = image_sym.normalise_plane([1.0, 0.5, 100])
        trans = image_sym.reflection_transform(plane)
        im_reflected = sitk.Resample(im, im, trans, sitk.sitkLinear, 0.0, im.GetPixelID())
        im = im + im_reflected

        # Initialise the plane as something different and try to fit
        plane_init = [0.80, 0.7, 0.22]
        plane_init = image_sym.normalise_plane(plane_init)
        plane_est = image_sym.plane_of_reflection(im, plane_init, levels=[4])
        print('Initial plane:   ', plane_init)
        print('Estimated plane: ', plane_est)
        print('True plane:      ', plane)

        # Show the result
        f = pl.figure()
        pl.imshow(sitk.GetArrayFromImage(im),
                  cmap = 'gray',
                  origin='lower',
                  extent = (-arr.shape[1]/2, arr.shape[1]/2, -arr.shape[0]/2, arr.shape[0]/2))
        x = np.linspace(-arr.shape[1]/2, arr.shape[1]/2, 100)
        y = image_sym.plane_2d(x, plane)
        pl.plot(x, y, 'r-', label = "Truth")
        y_ = image_sym.plane_2d(x, plane_init)
        pl.plot(x, y_, 'b-', label = "Init.")
        y__ = image_sym.plane_2d(x, plane_est)
        pl.plot(x, y__, 'g--', label = "Est.")
        pl.plot((0, 0), (0, 0), 'ro')
        pl.xlim(-arr.shape[1]/2, arr.shape[1]/2)
        pl.ylim(-arr.shape[0]/2, arr.shape[0]/2)
        pl.legend(loc = 1)
        pl.show()
