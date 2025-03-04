import torch
import numpy as np
from PIL import Image
from diffusers import CogView4Pipeline

class CogView4Generator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "num_inference_steps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "generators"

    def __init__(self):
        self.pipe = None
        
    def load_model(self):
        if self.pipe is None:
            print("Carregando modelo CogView4...")
            # Carrega o modelo com otimizações de memória
            self.pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16)
            self.pipe.enable_model_cpu_offload()
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            print("Modelo CogView4 carregado com sucesso!")
    
    def _convert_pil_to_comfyui_tensor(self, pil_image, width, height):
        """
        Converte uma imagem PIL para um tensor no formato esperado pelo ComfyUI.
        Lida especificamente com o problema (1, 1, 1024).
        """
        # Certifique-se de que temos uma imagem RGB
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # Redimensiona se necessário para garantir as dimensões corretas
        if pil_image.size != (width, height):
            pil_image = pil_image.resize((width, height), Image.LANCZOS)
        
        # Converte para numpy array
        np_image = np.array(pil_image, dtype=np.float32) / 255.0
        
        # Verifica se o formato está correto (H, W, 3)
        if len(np_image.shape) != 3 or np_image.shape[2] != 3:
            print(f"AVISO: Formato incorreto detectado: {np_image.shape}")
            # Cria uma matriz vazia do tamanho correto
            np_image = np.zeros((height, width, 3), dtype=np.float32)
        
        # Converte para tensor no formato (C, H, W)
        tensor = torch.from_numpy(np_image).permute(2, 0, 1)
        
        return tensor
            
    def generate(self, prompt, width, height, num_inference_steps, guidance_scale, num_images, seed=None):
        self.load_model()
        
        # Configura seed para reprodutibilidade
        if seed is not None and seed > 0:
            generator = torch.Generator("cuda").manual_seed(seed)
        else:
            generator = None
            
        print(f"Gerando {num_images} imagem(ns) com CogView4...")
        print(f"Prompt: {prompt}")
        
        try:
            # Gera as imagens
            output = self.pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                generator=generator
            )
            
            # Cria uma lista de tensores vazios como fallback
            fallback_tensor = torch.zeros((3, height, width), dtype=torch.float32)
            fallback_tensors = [fallback_tensor] * num_images
            
            # Verifica se temos imagens na saída
            if not hasattr(output, 'images') or len(output.images) == 0:
                print("Aviso: Nenhuma imagem gerada pelo modelo. Usando fallback.")
                tensors = fallback_tensors
            else:
                # Tenta converter as imagens PIL para tensores
                tensors = []
                for i, pil_img in enumerate(output.images):
                    try:
                        # Usa a função específica para conversão
                        tensor = self._convert_pil_to_comfyui_tensor(pil_img, width, height)
                        tensors.append(tensor)
                    except Exception as e:
                        print(f"Erro ao converter imagem {i}: {str(e)}")
                        tensors.append(fallback_tensor)
                
                # Se não conseguimos converter nenhuma imagem, usa fallback
                if not tensors:
                    tensors = fallback_tensors
            
            # Combina em um tensor batch (resultado em formato channel-first: (N, 3, H, W))
            if len(tensors) > 1:
                result = torch.stack(tensors)
            else:
                result = tensors[0].unsqueeze(0)
            
            print(f"Tensor final (channel-first): forma={result.shape}, tipo={result.dtype}")
            
            # Verificação final de formato
            if result.shape[1] != 3 or len(result.shape) != 4:
                print(f"AVISO: Formato final incorreto: {result.shape}. Usando tensor vazio.")
                result = torch.zeros((num_images, 3, height, width), dtype=torch.float32)
            
            # Transpõe para o formato channel-last esperado pelo PIL (ou seja, (N, H, W, 3))
            result = result.permute(0, 2, 3, 1)
            print(f"Tensor final ajustado (channel-last): forma={result.shape}, tipo={result.dtype}")
            
            return (result,)
            
        except Exception as e:
            print(f"Erro ao gerar imagens com CogView4: {str(e)}")
            # Em caso de erro, retorna uma imagem preta
            black_image = torch.zeros((1, 3, height, width), dtype=torch.float32)
            return (black_image,)

# Registrar nós
NODE_CLASS_MAPPINGS = {
    "CogView4Generator": CogView4Generator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CogView4Generator": "CogView4 Generator"
}
