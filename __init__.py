import os
import sys

# Adiciona o diretório atual ao path do Python
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Agora importa o módulo
from cogview4_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']