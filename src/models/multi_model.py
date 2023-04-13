import tensorflow as tf
tfk = tf.keras

from .fkvae import fKVAE

class MultiModel(tfk.Model):
    def __init__(self, config, name='multi_model', **kwargs):
        super(MultiModel, self).__init__(self, name=name, **kwargs)
        self.config = config
        self.transversal_model = fKVAE(config, 'transversal_model', prefix='Tran')
        self.sagittal_model = fKVAE(config, 'sagittal_model', prefix='Sag')
        self.coronal_model = fKVAE(config, 'coronal_model', prefix='Cor')
    
    def parse_inputs(self, inputs):
        y_trans = inputs['transversal']
        y_sag = inputs['sagittal']
        y_cor = inputs['coronal']
        
        y_trans_ref = inputs['transversal_ref']
        y_sag_ref = inputs['sagittal_ref']
        y_cor_ref = inputs['coronal_ref']
        
        mask = tf.zeros((y_trans.shape[0], y_trans.shape[1]), dtype='bool')
        inputs_trans = {'input_video': y_trans, 'input_ref': y_trans_ref, 'input_mask': mask}
        inputs_sag = {'input_video': y_sag, 'input_ref': y_sag_ref, 'input_mask': mask}
        inputs_cor = {'input_video': y_cor, 'input_ref': y_cor_ref, 'input_mask': mask}
        return inputs_trans, inputs_sag, inputs_cor
        
    def call(self, inputs):
        inputs_trans, inputs_sag, inputs_cor = self.parse_inputs(inputs)

        self.transversal_model(inputs_trans)
        self.sagittal_model(inputs_sag)
        self.coronal_model(inputs_cor)

    @tf.function
    def eval(self, inputs):
        inputs_trans, inputs_sag, inputs_cor = self.parse_inputs(inputs)
        
        output_trans = self.transversal_model.eval(inputs_trans)
        output_sag = self.sagittal_model.eval(inputs_sag)
        output_cor = self.coronal_model.eval(inputs_cor)
        return output_trans, output_sag, output_cor

    def compile(self, num_batches, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, jit_compile=None, **kwargs):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.config.init_lr, 
                                                                     decay_steps=self.config.decay_steps*num_batches, 
                                                                     decay_rate=self.config.decay_rate, 
                                                                     staircase=True)        
        
        optimizer = tf.keras.optimizers.Adam(lr_schedule)

        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, **kwargs)