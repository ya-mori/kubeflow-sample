from kubeflow_pipeline.comands.argparse import command_args
from kubeflow_pipeline.comands.runner import compile_pipeline, execute_pipeline

if __name__ == '__main__':
    args = command_args()
    if args.mode == "compile":
        compile_pipeline(args.pipeline_name)
    elif args.mode == "run":
        execute_pipeline(args.pipeline_name, args.params)
    else:
        raise ValueError(f"no such mode : {args.mode}")
