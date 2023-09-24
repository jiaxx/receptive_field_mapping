import numpy as np
import scipy.ndimage as ni
import scipy.interpolate as ip
import matplotlib.pyplot as plt

def int2str(num,length=None):
    '''
    generate a string representation for a integer with a given length
    :param num: non-negative int, input number
    :param length: positive int, length of the string
    :return: string represetation of the integer
    '''

    rawstr = str(int(num))
    if length is None or length == len(rawstr):
        return rawstr
    elif length < len(rawstr):
        print('Length of the number is longer then defined display length!')
    elif length > len(rawstr):
        return '0'*(length-len(rawstr)) + rawstr


def sort_masks(masks, key_prefix=None, label_length=None):
    '''
    sort a dictionary of binary masks, big to small

    :param masks: dictionary of 2d binary masks
    :param key_prefix: str, prefix of keys in returned mask dictionary
    :param label_length: positive int, length of mask index string

    :return new_masks: dictionary of sorted 2d binary masks
    '''

    order = []

    for key, mask in masks.iteritems():
        order.append([key,np.sum(mask.flatten())])

    order = sorted(order, key=lambda a:a[1], reverse=True)

    new_masks = {}

    for i in range(len(order)):
        if key_prefix is not None:
            curr_key = key_prefix + '_' + int2str(i, label_length)
        else:
            curr_key = int2str(i, label_length)
        new_masks.update({curr_key : masks[order[i][0]]})
    return new_masks


def get_masks(labeled, min_area=None, max_area=None, is_sort=True, key_prefix = None, label_length=None):
    '''
    get mask dictionary from labeled map (labeled by scipy.ndimage.label function), masks with area smaller than
    minArea and max_area will be discarded.

    :param labeled: 2d array with non-negative int, labelled map (ideally the output of scipy.ndimage.label function)
    :param min_area: positive int, minimum area criterion of retained masks
    :param max_area: positive int, maximum area criterion of retained masks
    :param is_sort: bool, sort the masks by area or not
    :param key_prefix: str, the key prefix for returned dictionary
    :param label_length: positive int, the length of key index

    :return masks: dictionary of 2d binary masks
    '''

    mask_num = np.max(labeled.flatten())
    masks = {}
    for i in range(1, mask_num + 1):
        curr_mask = np.zeros(labeled.shape, dtype=np.uint8)
        curr_mask[labeled == i] = 1

        if min_area is not None and np.sum(curr_mask.flatten()) < min_area:
            continue
        elif max_area is not None and np.sum(curr_mask.flatten()) > max_area:
            continue
        else:
            if label_length is not None:
                mask_index = int2str(i, label_length)
            else:
                mask_index = str(i)

            if key_prefix is not None:
                curr_key = key_prefix + '_' + mask_index
            else:
                curr_key = mask_index
            masks.update({curr_key: curr_mask})

    if is_sort:
        masks = sort_masks(masks, key_prefix=key_prefix, label_length=label_length)

    return masks


def get_peak_weighted_roi(arr, thr):
    """
    get a weighted mask containing peak pixel

    :param arr: 2d array, could be binary or float, any non-zero, non-nan pixels will be considered as non-background
    :param thr: positive float, threshold to cut the arr into masks
    """

    nan_label = np.isnan(arr)
    arr2 = np.array(arr)
    arr2[nan_label] = np.nanmin(arr)
    labeled, _ = ni.label(arr2 >= thr)
    peak_coor = np.where(arr2==np.amax(arr2))
    peak_coor = tuple([coor[0] for coor in peak_coor])
    print(peak_coor)

    masks = get_masks(labeled=labeled, is_sort=False, label_length=None)

    for mask in masks.values():
        if mask[peak_coor] == 1:
            return arr2 * mask

    print('Threshold too high! No mask found. Returning None')
    return None


def interpolate(mask, ratio, method='cubic', fill_value=None):
    """
    increase spatial resolution by interpolating a mask

    :param mask: 2d array, input mask
    :param ratio: positive int, upsampling ratio
    :param method: str, interpolation method, {'linear', 'cubic', 'quintic'}. from scipy.interpolate.interp2d function
    :param fill_value: number, optional
                       If provided, the value to use for points outside of the interpolation domain.
                       If omitted (None), values outside the domain are extrapolated.
    :return: newMask: 2d array, interpolated mask
    """

    if not isinstance(ratio, int):
        print('interpolate ratio is not an integer. convert it into an integer.')
        ratio_int = int(round(ratio))
    else:
        ratio_int = ratio

    if ratio_int <= 1:
        raise ValueError('interpolate_rate should be an integer larger than 1!')

    row_labels = np.arange(mask.shape[0], dtype=np.float32)
    col_labels = np.arange(mask.shape[1], dtype=np.float32)

    row_labels_new = np.arange(0, mask.shape[0], 1. / ratio_int)
    col_labels_new = np.arange(0, mask.shape[1], 1. / ratio_int)

    mask_interpolation = ip.interp2d(col_labels, row_labels, mask, kind=method, fill_value=fill_value)

    new_mask = mask_interpolation(col_labels_new, row_labels_new)

    return new_mask


if __name__ == '__main__':

    #----------------------------------------------------
    mapp = (np.random.rand(10, 10) * 255).astype(np.uint8)
    peak_mask = get_peak_weighted_roi(mapp, 150)

    f = plt.figure(figsize=(15, 5))
    ax1 = f.add_subplot(121)
    fig1 = ax1.imshow(mapp, cmap='hot', vmin=0, vmax=255, interpolation='none')
    f.colorbar(fig1)
    ax2 = f.add_subplot(122)
    fig2 = ax2.imshow(peak_mask, cmap='hot', vmin=0, vmax=255, interpolation='none')
    f.colorbar(fig2)

    plt.show()
    # ---------------------------------------------------

    # ---------------------------------------------------
    mapp = np.zeros((10, 10))
    mapp[3, 4] = 1
    mapp[5, 8] = 3
    mapp_ip = interpolate(mapp, 10)

    print(mapp_ip.shape)

    f = plt.figure(figsize=(15, 5))
    ax1 = f.add_subplot(121)
    fig1 = ax1.imshow(mapp, cmap='hot', vmin=0, vmax=5, interpolation='none')
    f.colorbar(fig1)
    ax2 = f.add_subplot(122)
    fig2 = ax2.imshow(mapp_ip, cmap='hot', vmin=0, vmax=5, interpolation='none')
    f.colorbar(fig2)

    plt.show()
    # ---------------------------------------------------

    print('for debug ...')