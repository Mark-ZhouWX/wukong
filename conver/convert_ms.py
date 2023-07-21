import mindspore_lite as mslite
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_file", required=True)
    # parser.add_argument("--output_file", required=True)
    # parser.add_argument("--config_file", required=True)
    # opt = parser.parse_args()
    size = '512_512'
    # size = '512_640'
    # size = '640_384'
    # size = '640_512'
    model_file = f'../models/wukong_youhua_{size}_graph.mindir'
    output_file= f'../models/0719_out/out_wukong_youhua_{size}_graph'
    config_file = './config.cni'
    converter = mslite.Converter()
    converter.optimize = "ascend_oriented"
    # converter.convert(fmk_type=mslite.FmkType.MINDIR, model_file=opt.model_file, output_file=opt.output_file, config_file=opt.config_file)
    converter.convert(fmk_type=mslite.FmkType.MINDIR, model_file=model_file, output_file=output_file, config_file=config_file)

