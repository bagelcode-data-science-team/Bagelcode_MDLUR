import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Lambda
from bagelcode.ml_model.model import CategoricalDenseModel, WeightedSum, AutoEncoder, SkipAutoEncoder, Conv2dUnet, TransformerEncoder

class CUTAWAS(tf.keras.layers.Layer):
    
    def __init__(
        self,
        behavior_length,
        portrait_length,
        window_length,
        target_day,
        vocab_size_dict,
        **kwargs):
        
        self.behavior_length = behavior_length
        self.portrait_length = portrait_length
        self.window_length = window_length
        self.target_day = target_day
        self.vocab_size_dict = vocab_size_dict
        self.dense_layers = [128, 64]
        self.attention_num = 3
        self.trans_output_dim = 64
        
        super(CUTAWAS, self).__init__()

        user_model = self.build_user_model()
        portrait_input = Input((self.window_length, self.portrait_length))
        behavior_input = Input((self.window_length, self.behavior_length))
        portrait_static_model = self.build_portrait_static_model(portrait_input)
        portrait_ensemble_model = self.build_portrait_ts_model(portrait_input)
        behavior_ensemble_model = self.build_behavior_ts_model(behavior_input)

        concatenated = concatenate([
            user_model.output,
            portrait_static_model.output,
            portrait_ensemble_model.output, 
            behavior_ensemble_model.output
        ])

        denses = SkipAutoEncoder(
            init_channel_dim=32,
            depth=2,
            output_dim=self.target_day
        )
        
        z = denses(concatenated)

        self.full_model = Model(
            inputs=user_model.input + [portrait_input, behavior_input],
            outputs=z
        )
        
        
    def build_user_model(self):
        return CategoricalDenseModel()(self.vocab_size_dict)

    
    def build_portrait_static_model(self, input):
        portrait_static_input = Lambda(lambda x: x[:, -1,:])(input)
        portrait_static_model = AutoEncoder(16)
        portrait_static_output = portrait_static_model(portrait_static_input)
        return Model(inputs=input, outputs=portrait_static_output)
    
    
    def build_portrait_ts_model(self, input):
        portrait_conv_unet_model = self.get_conv_unet(self.portrait_length)
        portrait_transformer_model = self.get_transformer_encoder(self.portrait_length)
        portrait_ts_model_outputs = [model(input) for model in [portrait_conv_unet_model, portrait_transformer_model]]
        portrait_ensemble_output = WeightedSum()(portrait_ts_model_outputs)
        return Model(inputs=input, outputs=portrait_ensemble_output)    

    
    def build_behavior_ts_model(self, input):
        behavior_conv_unet_model    = self.get_conv_unet(self.behavior_length)
        behavior_transformer_model  = self.get_transformer_encoder(self.behavior_length)
        behavior_ts_model_outputs   = [model(input) for model in [behavior_conv_unet_model, behavior_transformer_model]]
        behavior_ensemble_output    = WeightedSum()(behavior_ts_model_outputs)
        return Model(inputs=input, outputs=behavior_ensemble_output)

    
    def get_transformer_encoder(self, feature_dim):
        return TransformerEncoder(
            self.window_length,
            feature_dim,
            self.dense_layers,
            self.trans_output_dim,
            add_time2vec=True,
            additional_dropout=False,
            attention_layer_num=self.attention_num,
        )
        

    def get_conv_unet(self, feature_dim):
        return Conv2dUnet(
            window_length=self.window_length,
            feature_dim=feature_dim,
            init_channel_dim=16,
            depth=2,
            output_dim=1
        )
        

    def call(self, inputs):
        return self.full_model(inputs)
      
        
    def get_config(self):
        
        config = super().get_config().copy()
        config.update({
            'behavior_length' : self.behavior_length,
            'portrait_length' : self.portrait_length,
            'window_length' : self.window_length,
            'target_day' : self.target_day,
            'vocab_size_dict' : self.vocab_size_dict,
            'dense_layers' : self.dense_layers,
            'attention_num' : self.attention_num,
            'trans_output_dim' : self.trans_output_dim,
        })
        return config
