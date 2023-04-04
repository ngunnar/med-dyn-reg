import tensorflow as tf
tfk = tf.keras

from .fkvae import fKVAE

class MultiModel(tfk.Model):
    def __init__(self, config, name='multi_model', **kwargs):
        super(MultiModel, self).__init__(self, name=name, **kwargs)
        self.config = config
        self.tranversal_model = fKVAE(config, 'tranversal_model', prefix='Tran')
        self.sagittal_model = fKVAE(config, 'sagittal_model', prefix='Sag')
        self.coronal_model = fKVAE(config, 'coronal_model', prefix='Cor')

    def call(self, inputs):
        y_tranversal = inputs['tranversal']
        y_sagittal = inputs['sagittal']
        y_coronal = inputs['coronal']
        mask = tf.zeros((y_tranversal.shape[0], y_tranversal.shape[1]), dtype='bool')

        self.tranversal_model({'input_video': y_tranversal, 'input_ref': y_tranversal[:,0,...], 'input_mask': mask})
        self.sagittal_model({'input_video': y_sagittal, 'input_ref': y_sagittal[:,0,...], 'input_mask': mask})
        self.coronal_model({'input_video': y_coronal, 'input_ref': y_coronal[:,0,...], 'input_mask': mask})

    @tf.function
    def eval(self, inputs):
        y_tranversal = inputs['tranversal']
        y_sagittal = inputs['sagittal']
        y_coronal = inputs['coronal']
        mask = tf.zeros((y_tranversal.shape[0], y_tranversal.shape[1]), dtype='bool')

        output_tran = self.tranversal_model.eval({'input_video': y_tranversal, 'input_ref': y_tranversal[:,0,...], 'input_mask': mask})
        output_sag = self.sagittal_model.eval({'input_video': y_sagittal, 'input_ref': y_sagittal[:,0,...], 'input_mask': mask})
        output_cor = self.coronal_model.eval({'input_video': y_coronal, 'input_ref': y_coronal[:,0,...], 'input_mask': mask})
        return output_tran, output_sag, output_cor

    def compile(self, num_batches, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, jit_compile=None, **kwargs):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.config.init_lr, 
                                                                     decay_steps=self.config.decay_steps*num_batches, 
                                                                     decay_rate=self.config.decay_rate, 
                                                                     staircase=True)        
        
        optimizer = tf.keras.optimizers.Adam(lr_schedule)

        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, **kwargs)