import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage.segmentation import find_boundaries
import matplotlib.gridspec as gsp
import os

def visualize_calcium_signal(image, reference_image=None, method='percentile', channel='green', **kwargs):
    """
    Visualizes single-channel calcium imaging data in red shades using various methods,
    with optional reference image for consistent scaling.
    
    Parameters:
    - image (numpy.ndarray): Single-channel input image
    - reference_image (numpy.ndarray, optional): Reference image for consistent scaling
    - method (str): Visualization method ('percentile', 'zscore', 'minmax', or 'adaptive')
    - kwargs: Additional parameters for specific methods
        - percentile_low, percentile_high: For 'percentile' method (default: 1, 99)
    
    Returns:
    - numpy.ndarray: RGB image with red channel visualization
    """
    # Ensure input is float
    img = image.astype(float)
    
    if method == 'percentile':
        p_low = kwargs.get('percentile_low', 1)
        p_high = kwargs.get('percentile_high', 99)
        
        # If reference image is provided, use its statistics for normalization
        if reference_image is not None:
            ref_img = reference_image.astype(float)
            low = np.percentile(ref_img, p_low)
            high = np.percentile(ref_img, p_high)
        else:
            low = np.percentile(img, p_low)
            high = np.percentile(img, p_high)
            
        # Apply normalization using reference statistics
        img = np.clip((img - low) / (high - low + 1e-9), 0, 1)
    
    # Create RGB image (rest of your existing code)
    rgb_img = np.zeros((*img.shape, 3))
    if channel == 'red':
        rgb_img[..., 0] = img
    elif channel == 'green':
        rgb_img[..., 1] = img
    
    return rgb_img


def get_labeled_masks_img(masks, roi_labels, cate):
    """
    Generates an RGB image highlighting the masks that belong to a specific category.

    Parameters:
    -----------
    masks : np.ndarray
        2D labeled mask of ROIs. 0 = background, and each ROI has a unique integer ID.
    roi_labels : np.ndarray
        1D array of labels corresponding to roi_ids, where label ∈ {-1, 0, 1}.
        -1 indicates excitatory, 0 unsure, 1 inhibitory.
    cate : int
        Category of interest. Must be one of {1, -1, 0}.
         -  1 → inhibitory
         - -1 → excitatory
         -  0 → unsure

    Returns:
    --------
    labeled_masks_img : np.ndarray
        3D image (height × width × 3), where pixels belonging to the specified `cate`
        are highlighted (by default, in the red channel).
    """
    # Create an empty RGB image (int32 to store 0..255 values if desired)
    labeled_masks_img = np.zeros(
        (masks.shape[0], masks.shape[1], 3), dtype='int32'
    )

    # Iterate over ROI IDs and their labels
    for roi_id, roi_label in enumerate(roi_labels):
        if roi_id == 0:
            # Skip the background ROI (id 0)
            continue
        if roi_label == cate:
            # Create a binary mask for this ROI
            roi_binary_mask = (masks == roi_id).astype('int32')

            # Convert to intensity 255 where ROI is present
            roi_binary_mask *= 255

            # Add it to the RED channel of our output image
            labeled_masks_img[:, :, 0] += roi_binary_mask

    return labeled_masks_img

def create_superimposed_mask_images(mean_func, max_func, masks, labels, mean_anat):
    """
    Generate superimposed mask images for visualization of functional and anatomical channels.

    Parameters
    ----------
    mean_func : np.ndarray
        Mean functional channel image (2D array).
    max_func : np.ndarray
        Max functional channel image (2D array).
    masks : np.ndarray
        2D labeled mask of ROIs. 0 = background, and each ROI has a unique integer ID.
    labels : np.ndarray
        1D array of labels corresponding to roi_ids, where label ∈ {-1, 0, 1}.
        -1 indicates excitatory, 0 unsure, 1 inhibitory.
    mean_anat : np.ndarray or None, optional
        Mean anatomical channel image (2D array), or None if not available.

    Returns
    -------
    mean_fun_channel : np.ndarray
        Visualized mean functional channel (RGB image).
    max_fun_channel : np.ndarray
        Visualized max functional channel (RGB image).
    superimpose_mask_func : np.ndarray
        RGB image of mean functional channel with superimposed ROI boundaries.
    superimpose_mask_anat : np.ndarray or None
        RGB image of mean anatomical channel with superimposed ROI boundaries, or None if mean_anat is None.
    """
    # Visualize the mean and max functional channels
    mean_fun_channel = visualize_calcium_signal(mean_func, method='percentile', channel='green')
    max_fun_channel = visualize_calcium_signal(max_func, method='percentile', channel='green')

    # Get ROI masks for each neuron type
    inhibitory_masks_img = get_labeled_masks_img(masks, labels, 1)
    excitatory_masks_img = get_labeled_masks_img(masks, labels, -1)
    unsure_masks_img = get_labeled_masks_img(masks, labels, 0)

    # Find boundaries for each neuron type
    boundaries_inhibitory = find_boundaries(inhibitory_masks_img[..., 0], mode='outer')
    boundaries_excitatory = find_boundaries(excitatory_masks_img[..., 0], mode='thick')
    boundaries_unsure = find_boundaries(unsure_masks_img[..., 0], mode='outer')

    # Create RGB images for boundaries
    boundary_image_inh = np.zeros_like(inhibitory_masks_img)
    boundary_image_inh[boundaries_inhibitory] = [1, 0, 1]  # Magenta for inhibitory

    boundary_image_exc = np.zeros_like(excitatory_masks_img)
    boundary_image_exc[boundaries_excitatory] = [0, 0, 1]  # Blue for excitatory

    boundary_image_uns = np.zeros_like(unsure_masks_img)
    boundary_image_uns[boundaries_unsure] = [1, 1, 1]      # White for unsure

    # Combine boundary images with the mean functional channel
    superimpose_mask_func = boundary_image_inh + boundary_image_exc + boundary_image_uns + mean_fun_channel

    # If anatomical channel is provided, visualize and superimpose boundaries
    if mean_anat is not None:
        mean_anat_channel = visualize_calcium_signal(mean_anat, method='percentile', channel='red')
        superimpose_mask_anat = mean_anat_channel + boundary_image_inh + boundary_image_exc + boundary_image_uns
    else:
        superimpose_mask_anat = None

    return mean_fun_channel, max_fun_channel, superimpose_mask_func, superimpose_mask_anat

def plot_fov_summary(mean_fun_channel, max_fun_channel, masks, superimpose_mask_func, mean_anat, superimpose_mask_anat, save_path=None):

    plt.figure(figsize=(45, 30))
    gs = gsp.GridSpec(2, 4)

    # first row for functional channel
    plt.subplot(gs[0, 0])
    plt.imshow(mean_fun_channel)
    plt.title("Mean Functional Channel (green colorized)")

    plt.subplot(gs[0, 1])
    plt.imshow(max_fun_channel)
    plt.title("Max Functional Channel (green colorized)")

    plt.subplot(gs[0, 2])
    plt.imshow(masks)
    plt.title("Masks")

    plt.subplot(gs[0, 3])
    plt.imshow(superimpose_mask_func)
    plt.title("Superimposed Mask + Mean Func\nInhibitory (magenta), Excitatory (blue), Unsure (white)")

    # second row for anatomical channel
    if (mean_anat is not None) and (superimpose_mask_anat is not None):
        plt.subplot(gs[1, 0])
        plt.imshow(mean_anat)
        plt.title("Mean Anatomical Channel (red colorized)")

        plt.subplot(gs[1, 3])
        plt.imshow(superimpose_mask_anat)
        plt.title("Superimposed Mask + Mean Anat\nInhibitory (magenta), Excitatory (blue), Unsure (white)")

    if save_path is not None:
        figures_dir = os.path.join(save_path, 'Figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        save_path = os.path.join(figures_dir, 'FOV_summary.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()