import os
import pdb as p
import yaml

lambdas = [0.1]
config_file = "configs/prolificdreamer_shape.yaml"
prompts=[ "a metal wristwatch",]

for prompt in prompts:
    obj_file= prompt.replace(' ','_')
    guide_sape=f'./load/shapes_obj/{obj_file}.obj'
    command = f"python launch.py --config {config_file} --train --gpu 1 system.prompt_processor.prompt='{prompt}' system.guide_shape={guide_sape} system.loss.lambda_shape={0.1}"
    os.system(command)