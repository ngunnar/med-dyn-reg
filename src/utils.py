import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf


def plot(data, title, max_i, axs):
    for k in range(0, max_i):
        if k==0:
            #axs[k].title.set_text(title)
            axs[k].set_ylabel(title)
        if data.shape[-1] == 2:
            axs[k].imshow(draw_hsv(data[k,...]))            
        else:           
            axs[k].imshow(data[k,...], cmap='gray')

def latent_plot(latents):
    x_vae = latents['x']#latents[6]
    
    if 'smooth_mean' not in latents:
        dims = x_vae.shape[2]
        t = np.arange(x_vae.shape[1])
        figure, axs = plt.subplots(1,dims, figsize=(6.4*dims, 4.8))
        for i in range(dims):
            axs[i].plot(t, x_vae[0,:,i],'--', label='x')
    else:
        x_mu_smooth = latents['smooth_mean']
        x_cov_smooth = latents['smooth_cov']
        x_mu_filt = latents['filt_mean']
        x_covs_filt = latents['filt_cov']
        x_mu_filt_pred = latents['pred_mean']
        x_covs_filt_pred = latents['pred_cov']
        

        std_smooth = tf.sqrt(tf.linalg.diag_part(x_cov_smooth[0,...]))
        std_filt = tf.sqrt(tf.linalg.diag_part(x_covs_filt[0,...]))
        std_pred = tf.sqrt(tf.linalg.diag_part(x_covs_filt_pred[0,...]))
        t = np.arange(std_smooth.shape[0])
        dims = std_smooth.shape[1]
        figure, axs = plt.subplots(1,dims, figsize=(6.4*dims, 4.8))
        for i in range(dims):
            mu_s = x_mu_smooth[0,:,i]
            stf_s = std_smooth[:,i]
            mu_f = x_mu_filt[0,:,i]
            stf_f = std_filt[:,i]
            mu_p = x_mu_filt_pred[0,:,i]
            stf_p = std_pred[:,i]
            
            axs[i].plot(t, x_vae[0,:,i],'--', label='x')
            axs[i].plot(t, mu_s, 'r', label='x(t|T)')
            axs[i].fill_between(t, mu_s-stf_s, mu_s+stf_s, alpha=0.2, color='r')
            axs[i].plot(t, mu_f, 'g', label='x(t|t)')
            axs[i].fill_between(t, mu_f-stf_f, mu_f+stf_f, alpha=0.2, color='g')
            axs[i].plot(t, mu_p, 'y', label='x(t+1|t)')
            axs[i].fill_between(t, mu_p-stf_p, mu_p+stf_p, alpha=0.2, color='y')
            axs[i].legend(loc="upper left", ncol=1)
                    
    
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

def single_plot_to_image(y, y_hat, idx = 0):    
    figure, axs = plt.subplots(2,1, sharex=True, sharey=True)
    axs = axs.flatten()
    [ax.axis('off') for ax in axs]
    #plot(y[0,...], 'True image', 1, axs[0:1])
    #plot(y_hat[0,...], 'VAE', 1, axs[1:])
    axs[0].title.set_text('True image')        
    axs[0].imshow(y[0,idx,...], cmap='gray')
    axs[1].title.set_text('VAE')        
    axs[1].imshow(y_hat[0,idx,...], cmap='gray')
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

def draw_hsv(flow):
    h, w, c = flow.shape
    fx, fy = flow[...,0], flow[...,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def plot_to_image(y, data_arg):    
    step = 5
    s = 0
    l_org = y.shape[1] // step
    l = l_org
    figure, axs = plt.subplots(len(data_arg)+1,l, sharex=True, sharey=True, figsize=(40,10))
    axs = axs.flatten()
    [ax.axes.xaxis.set_ticks([]) for ax in axs]
    [ax.axes.yaxis.set_ticks([]) for ax in axs]
    
    if y is not None:
        plot(y[0,::step,...], 'True image', l_org, axs[s:l])
        s = l
        l = l+l_org
    
    for d in data_arg:
        plot(d['data'][0,::step,...].numpy(), d['name'].numpy().decode("utf-8"), l_org, axs[s:l])
        #plot(d['data'][0,::step,...].numpy(), d['name'], l_org, axs[s:l])
        s = l
        l = l+l_org
    
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