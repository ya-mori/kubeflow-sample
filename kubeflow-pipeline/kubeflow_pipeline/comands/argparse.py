from argparse import Action, ArgumentParser, Namespace


class ParamProcessor(Action):
    """
    ref : https://qiita.com/Hi-king/items/de960b6878d6280eaffc
    ex)
    --param aaa=bbb --param ccc=ddd
    """

    def __call__(self, parser, namespace, values, option_strings=None):
        param_dict = getattr(namespace, self.dest, [])
        if param_dict is None:
            param_dict = {}

        k, v = values.split("=")
        param_dict[k] = v
        setattr(namespace, self.dest, param_dict)


def command_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["compile", "run"])
    parser.add_argument("--pipeline_name", type=str, required=True)
    parser.add_argument(
        "--params", action=ParamProcessor, required=False, default=dict()
    )
    return parser.parse_args()
