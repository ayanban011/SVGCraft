# SVGCraft

## Description
Pytorch implementation of the paper [SVGCraft: Beyond Single Object Text-to-SVG Synthesis with Comprehensive Canvas Layout](https://arxiv.org/abs/2404.00412). For more information, Please check at: https://svgcraf.github.io

<img src="./images/CraftSVG.png"  alt="1" width = 1000px height = 500px >

## Getting Started 

### Step 1: Clone this repository and change directory to repository root
```bash
git clone https://github.com/ayanban011/SVGCraft.git 
cd SVGCraft
```

### Step 2: Setup and activate the conda environment with required dependencies
```bash
conda create -n svgcraft python=3.10 anaconda
conda activate svgcraft

# For diffusion
pip install -r requirements.txt

# For Abstraction
pip install -r requirements_lama.txt
pip install -r requirements.txt
```
Also install the [diffvg](https://github.com/BachiLi/diffvg) library by following the instructions at the corresponding github.


### Step 3: For bounding box generation
```bash
python prompt_batch.py --prompt-type demo --model gpt-3.5 --always-save --template_version v0.1
python scripts/eval_stage_one.py --prompt-type lmd --model gpt-3.5 --template_version v0.1
```

### Step 4: Layout to image generation
```bash
python generate.py --prompt-type demo --model gpt-4 --save-suffix "gpt-3.5" --repeats 5 --frozen_step_ratio 0.5 --regenerate 1 --force_run_ind 0 --run-model lmd_plus --no-scale-boxes-default --template_version v0.1
```

### Step 5: SVG generation
```bash
# detailed sketch
python painterly_rendering_strokes.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 500 --use_lpips_loss

# primitive shapes
python painterly_rendering_primtives.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 500 --use_lpips_loss

# CLIPArt
python painterly_rendering_color.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 500 --use_lpips_loss
```

### Step 6: SVG Abstraction
download the U2Net weights
```bash
wget https://huggingface.co/akhaliq/CLIPasso/resolve/main/u2net.pth --output-document=U2Net_/saved_models/u2net.pth
```
add lama
```bash
git clone https://github.com/advimman/lama.git
curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip
```
finally abstraction
```bash
python preprocess_images.py
python scripts/run_all.py --im_name "baboon"
```


## Citation

If you find this useful for your research, please cite it as follows:

```bash
@article{banerjee2024svgcraft,
  title={SVGCraft: Beyond Single Object Text-to-SVG Synthesis with Comprehensive Canvas Layout},
  author={Banerjee, Ayan and Mathur, Nityanand and Llad{\'o}s, Josep and Pal, Umapada and Dutta, Anjan},
  journal={arXiv preprint arXiv:2404.00412},
  year={2024}
}
```

## Acknowledgement

Many thanks to these excellent opensource projects 
* [Diffvg](https://github.com/BachiLi/diffvg) 
* [LLM-grounded diffusion](https://github.com/TonyLianLong/LLM-groundedDiffusion)

## Conclusion
Thank you for your interest in our work, and sorry if there are any bugs.

