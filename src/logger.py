try:
    from termcolor import cprint
except ImportError:

    def cprint(text, color, attrs=[]):
        print(text)


class Log:
    def section(text):
        print("=" * 40)
        cprint(text, "white", attrs=["bold"])
        print("-" * 40)

    def subsection(text):
        cprint("> %s" % text, "white", attrs=["bold"])

    def info(text):
        cprint(" - %s" % text, "white")

    def warn(text):
        cprint(" !! %s" % text, "yellow")

    def success():
        cprint(" - ok\n", "green")
