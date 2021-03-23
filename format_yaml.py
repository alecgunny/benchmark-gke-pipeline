import argparse
import re


field_re = re.compile("{{ .Values.[a-zA-Z0-9]+? }}")
result_re = re.compile("(?<={{ .Values.)[a-zA-Z0-9]+?(?= }})")


def main(filename, **kwargs):
    def replace_fn(match):
        varname = result_re.search(match.group(0)).group(0)
        try:
            return str(kwargs[varname])
        except KeyError:
            raise ValueError(
                "No value provided for wildcard {}".format(
                    varname
                )
            )
    with open(filename, "r") as f:
        contents = field_re.sub(replace_fn, f.read())
    print(contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "filename",
        type=str,
        help="File to format"
    )

    flags, others = parser.parse_known_args()

    try:
        with open(flags.filename, "r") as f:
            fields = field_re.findall(f.read())
    except FileNotFoundError:
        raise ValueError(f"Couldn't find yaml file {flags.filename}")

    full_parser = argparse.ArgumentParser(parents=[parser])
    for field in map(result_re.search, fields):
        if field is None:
            continue
        try:
            full_parser.add_argument(
                "--" + field.group(0),
                type=str,
                required=True
            )
        except argparse.ArgumentError as e:
            if "conflicting option string" not in str(e):
                raise

    flags = full_parser.parse_args([flags.filename] + others)
    main(**vars(flags))
