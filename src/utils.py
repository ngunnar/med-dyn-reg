import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        A = tf.linalg.cholesky(B)
        return True, A
    except:
        return False, None

def get_cholesky(A):
    """Find the nearest positive-definite matrix to input

    A Python/Tensorflow port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    is_pd, A_cholesky = isPD(A)
    if is_pd:
        return A_cholesky
    
    # Compute symmetric part of A
    B = (A + tf.transpose(A, perm=[0,1,3,2]))/2
    # Compute spectral decomposition B = UFU^T
    s, u, V = tf.linalg.svd(B)
    S = tf.linalg.diag(s)
    H = tf.matmul(V, tf.matmul(S, V), transpose_a=True)
    A2 = (B+H)/2
    A3 = (A2 + tf.transpose(A2, perm=[0,1,3,2]))/2
    is_pd, A_cholesky = isPD(A3)
    if is_pd:
        return A_cholesky
    spacing = np.spacing(tf.norm(A))
    I = tf.eye(A.shape[-1])
    k = 1
    while True:
        print(k)
        is_pd, A_cholesky = isPD(A3)
        if is_pd:
            return A_cholesky
        mineig = tf.math.reduce_min(tf.math.real(tf.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

def single_plot_to_image(y, y_hat):    
    figure, axs = plt.subplots(2,1, sharex=True, sharey=True)
    axs = axs.flatten()
    [ax.axis('off') for ax in axs]
    plot(y[0,...], 'True image', 1, axs[0:1])
    plot(y_hat[0,...], 'VAE', 1, axs[1:])
    plt.tight_layout()
    
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def plot_to_image(y, y_filt, y_smooth, y_vae):    
    step = 5
    l = y.shape[1] // step
    figure, axs = plt.subplots(4,l, sharex=True, sharey=True, figsize=(40,10))
    axs = axs.flatten()
    [ax.axis('off') for ax in axs]
    
    plot(y[0,::step,...], 'True image', l, axs[0:l])
    plot(y_filt[0,::step,...], 'KVAE filt', l, axs[l:2*l])
    plot(y_smooth[0,::step,...], 'KVAE smooth', l, axs[l*2:3*l])
    plot(y_vae[0,::step,...], 'VAE', l, axs[3*l:])
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def plot(y, title, max_i, axs):
    for k in range(0, max_i):
        if k==0:
            axs[k].title.set_text(title)        
        axs[k].imshow(y[k,...], cmap='gray')
    
def A_to_image(model):
    eigenvalues = [e.numpy() for e in tf.linalg.eig(model.kf.kalman_filter.transition_matrix)[0]]
    
    figure, ax = plt.subplots(1,1, figsize=(10,10))
    ax.axis('off')
    circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
    ax.add_patch(circ)
    [plt.plot(e.real, e.imag, 'go--', linewidth=2, markersize=12) for e in eigenvalues]
    title = str(model.kf.kalman_filter.transition_matrix.numpy())+ '\n'+ str(eigenvalues)
    ax.title.set_text(title)  
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image