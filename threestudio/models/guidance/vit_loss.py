from torchvision.transforms import Resize
from torchvision import transforms
import torch
import torch.nn.functional as F
import yaml
import pdb as p
import threestudio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix


class VitExtractor:
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, device):
        self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            # self.layers_dict[key] = kwargs[key] if key in kwargs.keys() else []
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_feature_from_input(self, input_img):  # List([B, N, D])
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 16

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_head_num(self):
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def get_queries_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        q = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[0]
        return q

    def get_keys_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        k = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]
        return k

    def get_values_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        v = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[2]
        return v

    def get_keys_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map

#@threestudio.register("vit_loss")
class LossG(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.extractor = VitExtractor(model_name=cfg['dino_model_name'], device=device)

        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = Resize(cfg['dino_global_patch_size'], max_size=480)

        self.global_transform = transforms.Compose([global_resize_transform,
                                                    imagenet_norm
                                                    ])

        self.lambdas = dict(
            lambda_global_cls=cfg['lambda_global_cls'],
            lambda_global_ssim=0,
            lambda_entire_ssim=0,
            lambda_entire_cls=0,
            lambda_global_identity=0
        )

    def update_lambda_config(self, step):
        if step == self.cfg['cls_warmup']:
            self.lambdas['lambda_global_ssim'] = self.cfg['lambda_global_ssim']
            self.lambdas['lambda_global_identity'] = self.cfg['lambda_global_identity']

        if step % self.cfg['entire_A_every'] == 0:
            self.lambdas['lambda_entire_ssim'] = self.cfg['lambda_entire_ssim']
            self.lambdas['lambda_entire_cls'] = self.cfg['lambda_entire_cls']
        else:
            self.lambdas['lambda_entire_ssim'] = 0
            self.lambdas['lambda_entire_cls'] = 0
    def forward(self, rgb_BCHW, temp_img):
        #self.update_lambda_config(inputs['step'])  #这里没有采用数据增强之类的，所以不用
        losses = {}
        loss_G = 0
        #sp.set_trace()
        ##结构loss
        if self.lambdas['lambda_global_ssim'] > 0:
            losses['loss_global_ssim'] = self.calculate_global_ssim_loss(rgb_BCHW, rgb_BCHW)
            loss_G += losses['loss_global_ssim'] * self.lambdas['lambda_global_ssim']

        if self.lambdas['lambda_entire_ssim'] > 0:
            losses['loss_entire_ssim'] = self.calculate_global_ssim_loss(rgb_BCHW, rgb_BCHW)
            loss_G += losses['loss_entire_ssim'] * self.lambdas['lambda_entire_ssim']

        ##

        ##外观loss
        if self.lambdas['lambda_entire_cls'] > 0:
            losses['loss_entire_cls'] = self.calculate_crop_cls_loss(rgb_BCHW, temp_img)
            loss_G += losses['loss_entire_cls'] * self.lambdas['lambda_entire_cls']

        if self.lambdas['lambda_global_cls'] > 0:
            losses['loss_global_cls'] = self.calculate_crop_cls_loss(rgb_BCHW, temp_img)
            loss_G += losses['loss_global_cls'] * self.lambdas['lambda_global_cls']

        ##

        # if self.lambdas['lambda_global_identity'] > 0:   #这一项是何作用
        #     losses['loss_global_id_B'] = self.calculate_global_id_loss(outputs['y_global'], inputs['B_global'])
        #     loss_G += losses['loss_global_id_B'] * self.lambdas['lambda_global_identity']

        losses['loss'] = loss_G
        return losses
    

    # def forward(self, outputs, inputs):
    #     #self.update_lambda_config(inputs['step'])  #这里没有采用数据增强之类的，所以不用
    #     losses = {}
    #     loss_G = 0
    #     #sp.set_trace()
    #     ##结构loss
    #     if self.lambdas['lambda_global_ssim'] > 0:
    #         losses['loss_global_ssim'] = self.calculate_global_ssim_loss(outputs['x_global'], inputs['A_global'])
    #         loss_G += losses['loss_global_ssim'] * self.lambdas['lambda_global_ssim']

    #     if self.lambdas['lambda_entire_ssim'] > 0:
    #         losses['loss_entire_ssim'] = self.calculate_global_ssim_loss(outputs['x_entire'], inputs['A'])
    #         loss_G += losses['loss_entire_ssim'] * self.lambdas['lambda_entire_ssim']

    #     ##

    #     ##外观loss
    #     if self.lambdas['lambda_entire_cls'] > 0:
    #         losses['loss_entire_cls'] = self.calculate_crop_cls_loss(outputs['x_entire'], inputs['B_global'])
    #         loss_G += losses['loss_entire_cls'] * self.lambdas['lambda_entire_cls']

    #     if self.lambdas['lambda_global_cls'] > 0:
    #         losses['loss_global_cls'] = self.calculate_crop_cls_loss(outputs['x_global'], inputs['B_global'])
    #         loss_G += losses['loss_global_cls'] * self.lambdas['lambda_global_cls']

    #     ##

    #     if self.lambdas['lambda_global_identity'] > 0:
    #         losses['loss_global_id_B'] = self.calculate_global_id_loss(outputs['y_global'], inputs['B_global'])
    #         loss_G += losses['loss_global_id_B'] * self.lambdas['lambda_global_identity']

    #     losses['loss'] = loss_G
    #     return losses

    def calculate_global_ssim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.global_transform(a) #from U-Net
            b = self.global_transform(b) #from U-Net
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def calculate_crop_cls_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(outputs, inputs):  # avoid memory limitations
            #p.set_trace()
            a = self.global_transform(a).unsqueeze(0).to(device) #from U-Net
            #p.set_trace()
            b = self.global_transform(b).unsqueeze(0).to(device) #from U-Net
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            with torch.no_grad():
                target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

    def calculate_global_id_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                keys_a = self.extractor.get_keys_from_input(a.unsqueeze(0), 11)
            keys_b = self.extractor.get_keys_from_input(b.unsqueeze(0), 11)
            loss += F.mse_loss(keys_a, keys_b)
        return loss
