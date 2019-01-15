# Supervised OIE Wrapper

This is thin wrapper over [AllenNLP](allennlp.org)'s [pretrained Open IE model](https://demo.allennlp.org/open-information-extraction).

Outputs predictions identical to those in the onlline demo, with batched gpu options.

## Install prerequisites

* AllenNLP
* docopt
* tqdm

Use the following to install requirements:

    pip install -r requirements.txt

## Run on raw sentences

    cd src
    python run_oie.py --in=path/to/input/file  --batch-size=<batch-size> --out=path/to/output/file [--cuda-device=<cude-device-identifier]
    
If `--cuda-device` is not specified, the model will run on the cpu.

## Input format

Raw sentences, each in a new line.

## Output Format
Each line pertains to a single OIE extraction:

    tokenized sentence <tab> ARG0:.. <tab> V:... <tab> ARG1:...  ...

## Example

See example of input and output files in [src/example.txt](src/example.txt) and [src/example.oie](src/example.oie).
