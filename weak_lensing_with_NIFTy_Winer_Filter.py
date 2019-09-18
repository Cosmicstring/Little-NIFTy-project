import galsim
import numpy as np
import matplotlib.pyplot as plt
import nifty5 as ift

from nifty5 import LinearOperator
# import pdb; pdb.set_trace()

class ShearOperator(LinearOperator):

	def __init__(self, domain, g1,g2):

		self._domain = ift.DomainTuple.make(domain)
		self._g1 = g1
		self._g2 = g2
		"""
		I have here to return back to the same field domain I started with,
		as I would do with a weak lensing transformation usually
		"""
		self._target = ift.DomainTuple.make(domain)
		self._capability = self.TIMES | self.ADJOINT_TIMES

	def apply(self, x, mode):

		self._check_mode(mode)

		if mode == self.TIMES:
		#Here the ordinary applycation of the shear_operator
			array = x.to_global_data()
			sheared = custom_shear(array, self._g1,self._g2)

			return ift.from_global_data(self._target, sheared)

		if mode == self.ADJOINT_TIMES:
		# Here the adjoint is just the inverse, since I have
		# a unitary operator
			array = x.to_global_data()
			inverse_sheared = custom_shear(array, -self._g1,-self._g2)
			return ift.from_global_data(self._target, inverse_sheared)

def custom_shear(original, g1,g2):
	
	g = pow(g1**2 + g2**2,0.5)
	A = np.array([[ 1.+ g1,  -g2  ],
                 [  -g2 , 1. - g1 ]]) / np.sqrt(1.-g**2)

	dim_x, dim_y = original.shape[0], original.shape[1]
	x_center, y_center = dim_x / 2, dim_y / 2
	result = np.zeros((dim_x, dim_y))
	for x in range(dim_x):
		for y in range(dim_y):
			A_dot_old = A.dot([float(x-x_center),float(y-y_center)])
			xp, yp = int(A_dot_old[0]+x_center), int(A_dot_old[1]+y_center)
			if 0 <= xp < dim_x and 0 <= yp < dim_y:
				result[x,y] = original[xp,yp]
	return result

def test_shearing(gal_flux, gal_r0, g1, g2, pixel_scale):

	gal = galsim.Exponential(flux = gal_flux, scale_radius = gal_r0)
	image_original = gal.drawImage(scale = pixel_scale)

	gal_shear = gal.shear(g1=g1, g2=g2)
	image_galsim_shear = gal_shear.drawImage(scale = pixel_scale)

	image_custom = np.asarray(image_original.array)
	image_custom_shear = custom_shear(image_custom, g1,g2)
	image_custom_unshear = custom_shear(image_custom_shear, -g1,-g2)

	f, axarr = plt.subplots(2,2)
	axarr[0,0].imshow(image_original.array)
	axarr[0,1].imshow(image_custom_unshear)
	axarr[1,0].imshow(image_galsim_shear.array)
	axarr[1,1].imshow(image_custom_shear)

	plt.show()

gal_flux = 1.e7   # counts
gal_r0 = 10       # arcsec
g1 = 0.5          #
g2 = 0.1           #
pixel_scale = 0.15  # arcsec / pixel

"""
Make the ground_truth image from gal, and define the position space
"""
gal = galsim.Gaussian(flux = gal_flux, scale_radius = gal_r0)
image_original = gal.drawImage(scale = pixel_scale)
image_sheared_galsim = i

plt.imshow(image_original.array)
plt.show()
exit()

image_original_arr = np.asarray(image_original.array)

position_space = ift.RGSpace(image_original_arr.shape)

ground_truth = ift.from_global_data(position_space, image_original_arr)

# It would be better to enforce the positivity of my prior spectrum
# by using lognormal
prior_spectrum = lambda k: 5/(10. + k**2.)	
"""
Make the data, which is the sheared image, with known shear
and the shear operator which would be my instrument response
"""

R = ShearOperator(position_space, g1, g2)

data_space = R.target

sheared_image = custom_shear(image_original_arr, g1=g1,g2=g2)
"""
Now the Wiener filter part
"""

N = ift.ScalingOperator(0.1, data_space)
data = ift.from_global_data(data_space, sheared_image) + N.draw_sample()

# Similar to the fourier transform, but we have real numbers as output
harmonic_space = position_space.get_default_codomain()
HT = ift.HartleyOperator(harmonic_space, target=position_space) 

S_h = ift.create_power_operator(harmonic_space, prior_spectrum)

# @ is as if doing function composition 
S = HT @ S_h @ HT.adjoint

D_inv = S.inverse + R.adjoint @ N.inverse @ R
j = (R.adjoint @ N.inverse)(data)

IC = ift.GradientNormController(name = 'CG', iteration_limit=100, tol_abs_gradnorm=1e-7)

D = ift.InversionEnabler(D_inv.inverse, IC, approximation=S)

# Conjugate gradiend applied here
m = D(j)

result_image = m.val

# Plot the images
f, axarr = plt.subplots(1,3)
axarr[0].imshow(image_original_arr)
axarr[1].imshow(data.val)
axarr[2].imshow(m.val)

plt.savefig('Original_data_mean.png', dpi = 150)

# Plot the profile

m_arr = m.val
dim_x, dim_y = m_arr.shape
Y_dim = int(dim_y/2)
m_profile = m_arr[:, Y_dim]

x = np.arange(0, dim_x)

f, axarr = plt.subplots(1,2)
axarr[0].plot(x, image_original_arr[:, Y_dim])
axarr[1].plot(x, m_profile)

plt.savefig('Profile.png', dpi = 150)


# We need this because from a chained operator (the one defined above on line 33, is not clear how to draw samples)
S = ift.SandwichOperator.make(HT.adjoint, S_h)

# Draw a sample from a Gaussian with zero mean and variance with variance of S
ground_truth2 = S.draw_sample()

f, axarr = plt.subplots(1,3)
axarr[0].imshow(image_original_arr)
axarr[1].imshow(data.val)
axarr[2].imshow(ground_truth2.val)

plt.savefig('Ground_truth_1_vs_2.png', dpi=150)

D = ift.WienerFilterCurvature(R, N, S, IC, IC).inverse
N_samples = 5
samples = [D.draw_sample() + m for i in range(N_samples)]

# Plot the estimated uncertainty
f, axarr = plt.subplots(2,1)

x = np.arange(0, dim_x)
axarr[0].plot(x, image_original_arr[:, Y_dim])

for i in range(N_samples):
	m_arr = samples[i].val
	dim_x, dim_y = m_arr.shape
	m_profile = m_arr[:, Y_dim] - image_original_arr[:, Y_dim]

	axarr[1].plot(x, m_profile)

plt.savefig('Profile_with_error.png', dpi = 150)
