import os, time, cv2, math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
from skimage.morphology import disk
from skimage.filters import median
# check for dependencies
home_dir = os.path.dirname(os.path.realpath(__file__))

def load_file( file ):
	print( "loading 'C3S-Balken {}'".format( file['filename'] ) )

	with open( home_dir + os.sep + file['filename'] ) as f_handle:
		for x in range(15):
			line = next(f_handle)
			if line.__contains__(' = '):
				d = line.split(' = ', 1)
				if   d[0].__contains__('z-unit'):	file['unit']	 = d[1].replace('[', '').replace(']', '').replace('\n', '')
				elif d[0].__contains__('x-pixels'):	file['x-pixels'] = int(d[1])
				elif d[0].__contains__('y-pixels'):	file['y-pixels'] = int(d[1])
				elif d[0].__contains__('x-length'):	file['x-length'] = int(d[1])
		file['scale'] = file['x-length']/file['x-pixels']

	file['data']	= np.flipud( np.loadtxt( home_dir + os.sep + file['filename'], delimiter="\t", dtype=str ) ) # for some reason, the dataset is flipped
	file['data']	= file['data'][0:file['y-pixels'],0:file['x-pixels']].astype(np.float64)
	file['min']		= np.min( file['data'])
	file['max']		= np.max( file['data'])
	file['mean']	= np.mean(file['data'])
	#file['data']	= np.clip(file['data'], file['clip_min'], file['clip_max'])

	# get a thumbnail in 8 bit
	resize_factor = 1
	file['thumb']	= cv2.resize(np.clip(file['data'], file['clip_min'], file['clip_max']),(int(file['x-pixels']/resize_factor), int(file['y-pixels']/resize_factor)),interpolation=cv2.INTER_AREA)

	plot_image(file['thumb'], file['filename'], file['scale'])

	file['thumb']	= cv2.normalize( (file['thumb']+(file['thumb'].min()*-1)), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

	print( 'x/y-scale: {:.3f} {}/px'.format(file['scale'], file['unit']) )
	print( 'dataset size: {:.1f} x {:.1f} {} ({} x {} px)'.format( file['x-pixels']*file['scale'], file['y-pixels']*file['scale'], file['unit'], file['x-pixels'], file['y-pixels'] ) )
	print('Min: {}, Max: {}, Mean: {:.6f}'.format( int(file['min']), int(file['max']), file['mean'] ))


	cv2.imwrite( home_dir + os.sep + file['filename'] + '.tif', (file['data']+(file['min']*-1)).astype(np.uint16) )

	return file

def get_ticks_in_mm( factor_nm_per_px, distance=1, low_label_limit=0, high_limit=100 ):
	ticks  = []
	labels = []
	for n in range(low_label_limit, high_limit+1, distance):
		labels.append(n)#*1000*1000
		ticks.append(n*1000*1000/factor_nm_per_px)

	return ticks, labels

def filter_label( ticks, labels, max_width ):
	ticks_filtered  = []
	labels_filtered = []
	for i, t in enumerate(ticks):
		if t <= max_width:
			ticks_filtered.append(t)#*1000*1000
			labels_filtered.append(labels[i])

	return ticks_filtered, labels_filtered

def align_images(img1, img2):
	sift = cv2.SIFT_create()
	#  detect SIFT features in both images
	keypoints_1, descriptors_1 = sift.detectAndCompute(cv2.GaussianBlur(img1, (7, 7), 0),None)
	keypoints_2, descriptors_2 = sift.detectAndCompute(cv2.GaussianBlur(img2, (7, 7), 0),None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(descriptors_1,descriptors_2,k=2)

	good = []
	for m,n in matches:
		if m.distance < 0.4*n.distance: #modify factor 0.2 to improve or worsen the good matches
			good.append(m)
	matches = good
	print('Found {} matching point pairs'.format(len(matches)))

	# draw first 50 matches
	matched_img = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches, img2, flags=2)

	plt.figure(figsize = (20,6))
	plt.title("visualized matches")
	plt.imshow( np.rot90(matched_img), cmap='gray' )
	plt.show()

	# allocate memory for the keypoints (x, y)-coordinates from the
	# top matches -- we'll use these coordinates to compute our
	# homography matrix
	ptsA = np.zeros((len(matches), 2), dtype="float")
	ptsB = np.zeros((len(matches), 2), dtype="float")
	# loop over the top matches
	for (i, m) in enumerate(matches):
		# indicate that the two keypoints in the respective images
		# map to each other
		ptsA[i] = keypoints_1[m.queryIdx].pt
		ptsB[i] = keypoints_2[m.trainIdx].pt

	# compute the homography matrix between the two sets of matched
	# points
	(H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC, ransacReprojThreshold=1 )
	# use the homography matrix to align the images
	(h, w) = img1.shape[:2]

	return H, cv2.warpPerspective(img1, H, (w, h))

def get_value_at_point( img, p=(0,0) ):
	p = (p[0], p[1], img[p[0],p[1]])
	print( 'new point', int(p[0]), int(p[1]) )
	return p

def get_mean_value_At_point( img, p=(0,0), buffer=5 ):
	area = img[p[0]-buffer:p[0]+buffer,p[1]-buffer:p[1]+buffer]
	p = (int(p[0]), int(p[1]), area.mean())
	print( 'new point',int(p[0]), int(p[1]), int(p[2]),'std dev:', area.std() )
	return p

#https://gist.github.com/ryanbranch/8aa3f0768c6cb9268296468d63f8f21c
def computeBestFitPlane(points):
    valType = np.float64

    n = points.shape[0]  # n: the integer number of (X, Y, Z) coordinate triples in points
    if n < 3:
        return None

    # Determination of (X, Y, Z) coordinates of centroid ("average" point along each axis in dataset)
    sum = np.zeros((3), dtype=valType)
    for p in points:
        sum += p
    centroid = sum * (1.0 / valType(n))

    # Uses Emil Ernerfeldt's technique to calculate the full 3x3 covariance matrix, excluding symmetries
    xx = 0.0
    xy = 0.0
    xz = 0.0
    yy = 0.0
    yz = 0.0
    zz = 0.0
    for p in points:
        r = p - centroid
        xx += r[0] * r[0]
        xy += r[0] * r[1]
        xz += r[0] * r[2]
        yy += r[1] * r[1]
        yz += r[1] * r[2]
        zz += r[2] * r[2]
    xx /= valType(n)
    xy /= valType(n)
    xz /= valType(n)
    yy /= valType(n)
    yz /= valType(n)
    zz /= valType(n)

    weighted_dir = np.zeros((3), dtype=valType)
    axis_dir = np.zeros((3), dtype=valType)

    # X COMPONENT
    det_x = (yy * zz) - (yz * yz)
    axis_dir[0] = det_x
    axis_dir[1] = (xz * yz) - (xy * zz)
    axis_dir[2] = (xy * yz) - (xz * yy)
    weight = det_x * det_x
    if np.dot(weighted_dir, axis_dir) < 0.0:
        weight *= -1.0
    weighted_dir += axis_dir * weight

    # Y COMPONENT
    det_y = (xx * zz) - (xz * xz)
    axis_dir[0] = (xz * yz) - (xy * zz)
    axis_dir[1] = det_y
    axis_dir[2] = (xy * xz) - (yz * xx)
    weight = det_y * det_y
    if np.dot(weighted_dir, axis_dir) < 0.0:
        weight *= -1.0
    weighted_dir += axis_dir * weight

    # Z COMPONENT
    det_z = (xx * yy) - (xy * xy)
    axis_dir[0] = (xy * yz) - (xz * yy)
    axis_dir[1] = (xy * xz) - (yz * xx)
    axis_dir[2] = det_z
    weight = det_z * det_z
    if np.dot(weighted_dir, axis_dir) < 0.0:
        weight *= -1.0
    weighted_dir += axis_dir * weight

    a = weighted_dir[0]
    b = weighted_dir[1]
    c = weighted_dir[2]
    d = np.dot(weighted_dir, centroid) * -1.0  # Multiplication by -1 preserves the sign (+) of D on the LHS
    normalizationFactor = math.sqrt((a * a) + (b * b) + (c * c))
    if normalizationFactor == 0:
        return None
    elif normalizationFactor != 1.0:  # Skips normalization if already normalized
        a /= normalizationFactor
        b /= normalizationFactor
        c /= normalizationFactor
        d /= normalizationFactor
    # Returns a float 4-tuple of the A/B/C/D coefficients such that (Ax + By + Cz + D == 0)
    return (a, b, c, d)

def get_min_dim(im1, im2):
	a = im2.shape[0]
	if im1.shape[0] < a:
		a = im1.shape[0]

	b = im2.shape[1]
	if im1.shape[1] < b:
		b = im1.shape[1]

	return a, b

# https://stackoverflow.com/questions/53698635/how-to-define-a-plane-with-3-points-and-plot-it-in-3d
def get_plane( difference, p0, p1, p2, buffer = 5 ):
	p0 = get_mean_value_At_point( difference, p0, buffer )
	p1 = get_mean_value_At_point( difference, p1, buffer )
	p2 = get_mean_value_At_point( difference, p2, buffer )

	x0, y0, z0 = p0
	x1, y1, z1 = p1
	x2, y2, z2 = p2

	points = [p0,p1 ,p2 ]

	ux, uy, uz = [x1-x0, y1-y0, z1-z0]
	vx, vy, vz = [x2-x0, y2-y0, z2-z0]

	u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

	point  = np.array(p0)
	normal = np.array(u_cross_v)

	d = -point.dot(normal)

	xx, yy = np.meshgrid(range(difference.shape[0]), range(difference.shape[1]))

	print(normal[0], normal[1], normal[2], d)
	z = np.array( (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2] )

	# plot the surface
	#fig = plt.figure(figsize = (10,4))
	#plt.title("background 3-point plane orientation")
	#ax = plt.axes(projection='3d')
	#ax.plot3D(xx, yy, z)
	#ax.plot(x0, y0, z0, marker='o', color="red")
	#ax.plot(x1, y1, z1, marker='o', color="red")
	#ax.plot(x2, y2, z2, marker='o', color="red")
	#plt.show()

	plt.figure(figsize = (20,7))
	plt.title("calculated 3-point background plane")
	plt.imshow( z )#, cmap='gray'
	plt.plot( x0, y0, marker='o', color="red" )
	plt.plot( x1, y1, marker='o', color="red" )
	plt.plot( x2, y2, marker='o', color="red" )
	plt.show()

	#z = np.rot90(z)

	return z, points

def get_multipoint_plane( difference, points2d, shape=(0,0) ):

    points = np.empty([len(points2d), 3])
    for i, p in enumerate(points2d):
        points[i] = get_mean_value_At_point( difference, p, 5 ) #get_value_at_point( difference, p )

    (a,b,c,d) = computeBestFitPlane(points)
    #print( a,b,c,d )

    X,Y = np.meshgrid(range(shape[0]), range(shape[1]))
    Z = (- a*X - b*Y - d) / c

    plt3d = plt.figure(figsize = (10,4))
    plt.title("background multipoint plane orientation")
    ax = plt3d.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z)
    for p in points:
        ax.plot(p[0], p[1], p[2], marker='o', color="red")
    plt.show()

    plt.figure(figsize = (20,7))
    plt.title("calculated multipoint background plane")
    plt.imshow( Z )#, cmap='gray'
    #for p in points:
    #    plt.plot(p[0], p[1], marker='o', color="red")
    plt.show()

    Z = np.rot90(Z)

    return Z, points

# mask_filename			- filename of the mask, which defines, which areas should be used to calculate the correction plane
# difference			- aligned difference image
# scale					- scale from a file object containing
# nth_point				- use the crossing point of every nth row/column to get points used to calculate the correction plane.
# median_filter_kernel	- use a median filter if the value is > 1. The filter will be processed on a smaller image, resizing is defined by nth_point
def get_background_correction(mask_filename, difference, scale, nth_point = 10, median_filter_kernel = 21):

	# load mask as boolean
	resin_mask = np.flipud(cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE))[0:difference.shape[0], 0:difference.shape[1]].astype(bool)

	if median_filter_kernel > 1:
		print( 'processing median')
		median_img = cv2.resize(difference, (int(difference.shape[1]/nth_point), int(difference.shape[0]/nth_point)))
		median_img = median(median_img, disk(median_filter_kernel))
		median_img = cv2.resize(median_img, (difference.shape[1], difference.shape[0]))
		m = np.ma.array( median_img, mask=resin_mask )
	else:
		print( 'modifying mask - removing very deviating pixels')
		# remove parts of the image with a large deviation.
		# This is fairly manual.
		fa = 2
		fb = 4 #1.90#1.059#0.01251#.5
		resin_difference = np.ma.array(difference, mask=resin_mask)
		rd_mean = np.mean(resin_difference)
		rd_std  = np.std(resin_difference)
		resin_mask = (resin_mask | np.invert( ( difference < (rd_mean+fa*rd_std) ) & ( difference > (rd_mean-fb*rd_std) ) ).astype(bool) )
		m = np.ma.array( difference, mask=resin_mask )

	plot_image(m, 'Resin-Mask', scale)

	# extract the points
	points = []
	for x, r in enumerate(difference):
		if x % nth_point == 0:
			for y,z in enumerate(r):
				if ( y % nth_point == 0 ) & ( not resin_mask[x,y] ):
					points.append( (y,x,z) )

	points = np.array( points )
	print("using {} points to extract the correction plane. Used every {}th point.".format(len(points), nth_point*nth_point))

	# finally calculate the correction plane
	(a,b,c,d) = computeBestFitPlane(points)

	X,Y = np.meshgrid(range(difference.shape[1]), range(difference.shape[0]))
	background = (- a*X - b*Y - d) / c

	plot_image(background, "Background", scale)
	plot_image(m-background, 'difference', scale)

	return resin_mask, background

img_nr = 0
def plot_image(image, title, scale, cmap = 'gray'):
	global img_nr
	if image.shape[0] > image.shape[1]: image = np.rot90(image)

	ticks_f, labels_f = get_ticks_in_mm( scale, distance=1, low_label_limit=0, high_limit=20 )

	# plot the background
	fig, axis = plt.subplots(1,1, figsize = ((30,6)))
	plot = axis.imshow( image, cmap=cmap )

	ticks, labels = filter_label( ticks_f, labels_f, image.shape[1] )
	axis.set_xticks(ticks, labels)
	axis.set_xlabel('x in mm')

	ticks, labels = filter_label( ticks_f, labels_f, image.shape[0] )
	axis.set_yticks(ticks, labels)
	axis.set_ylabel('y in mm')

	axis.set_title(title)

	clb = fig.colorbar(plot, cax=make_axes_locatable(axis).append_axes('right', size='2%', pad=0.05), orientation='vertical')
	clb.ax.set_title('Δz in nm')

	plt.tight_layout()
	plt.savefig("{} - {}.png".format(img_nr, title))
	img_nr += 1
	plt.show()


# https://github.com/scikit-image/scikit-image/blob/5da1d5800f53fcdb42202396b9219ce5e0579440/skimage/measure/profile.py#L130-L174
# not aliased!
def line_profile_coordinates(src, dst, linewidth=1):
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src
    theta = np.arctan2(d_row, d_col)

    length = int(np.ceil(np.hypot(d_row, d_col) + 1))
    # we add one above because we include the last point in the profile
    # (in contrast to standard numpy indexing)
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)

    # we subtract 1 from linewidth to change from pixel-counting
    # (make this line 3 pixels wide) to point distances (the
    # distance between pixel centers)
    col_width = (linewidth - 1) * np.sin(-theta) / 2
    row_width = (linewidth - 1) * np.cos(theta) / 2
    perp_rows = np.stack([np.linspace(row_i - row_width, row_i + row_width,
                                      linewidth) for row_i in line_row])
    perp_cols = np.stack([np.linspace(col_i - col_width, col_i + col_width,
                                      linewidth) for col_i in line_col])
    # changed output format to 2 1D-arrays containing all x and all y coordinates of the line-pixels
    return perp_rows.astype(int).flatten(), perp_cols.astype(int).flatten()#np.vstack((perp_rows.astype(int).flatten(), perp_cols.astype(int).flatten())).T


def draw_rect_on_fft_mask( p, mask, width=20, value=0 ):
    rows, cols = mask.shape

    d = int(width/2)

    # set [0,0] coordinate to the center of the fft result
    p = [int(rows/2)+p[0],int(cols/2)+p[1]]

    # set mask to value at opposing quadrants.
    mask[p[0]-d : p[0]+d, p[1]-d : p[1]+d] = value
    mask[(-p[0])-d : (-p[0])+d, (-p[1])-d : (-p[1])+d] = value

    return mask

def draw_line_on_fft_mask( p1, p2, mask, linewidth=5, value=0 ):
    rows, cols = mask.shape

    # set [0,0] coordinate to the center of the fft result
    p1 = [int(rows/2)+p1[0],int(cols/2)+p1[1]]
    p2 = [int(rows/2)+p2[0],int(cols/2)+p2[1]]

    # set mask to value at opposing quadrants.
    line_cx, line_cy = line_profile_coordinates(p1, p2, linewidth=linewidth)
    mask[line_cx, line_cy] = value
    mask[line_cx*-1, line_cy*-1] = value

    return mask

### actual program start
if __name__ == '__main__':
    print( "This libary is not callable!" )