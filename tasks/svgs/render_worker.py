"""Private renderer process: no API credentials and no external resource loading."""
import resource
import sys


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    from cairosvg.surface import PNGSurface

    def reject_resource(url, resource_type):
        raise ValueError('External resources are not allowed')

    svg = sys.stdin.buffer.read(1_000_001)
    if len(svg) > 1_000_000:
        raise ValueError('SVG too large')
    png = PNGSurface.convert(
        bytestring=svg, output_width=800, output_height=600,
        unsafe=False, url_fetcher=reject_resource,
    )
    sys.stdout.buffer.write(png)


if __name__ == '__main__':
    main()
