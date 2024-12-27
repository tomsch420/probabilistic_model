from typing_extensions import Dict, Any

circled_product = 'shape=stencil(tZXbboQgEIafhtsGIY3XjW3fg+psJcsCAbe7ffsiSFc8dduiMZqZYT5/BgcQrWzLNCCCW0SfESEFxu7p7MvEZlZD3QXngV+hCW7bGXWEC2+6AcBlC4Z3fZS+IPzkxvQ3rWolpSNwJW0SGcUdjHHpcvE1wIZvfyaWdvwTdGAGhcGLyOv92IfHfcDFTnpTbFSfXW8+cKq42GvhsoGLv2Np5TxrPzet3lh9fDfqLJt51mrsoAwsBL7DXIjQeBvTDANqJZRxjvD280CEYn9tl2nc2W47uHU42c7TrB84C8TwSX3Are4kXc9oLrMjQnA5QpQporwLkaoo/6+C/EaFX/yVOsXC56rw4ha6PTMQgmv7E2N6QEwPkKxT92mzrvDecIB5xxc=);whiteSpace=wrap;html=1;labelPosition=center;verticalLabelPosition=bottom;align=center;verticalAlign=top;'
circled_sum = 'shape=stencil(tZVtb4MgEMc/DW8XHmLSt43bvgdTOkkpEKBr9+2LoGvxgXULGqPh/tzPu5MDQGrbUc0Ahh0grwBjBKF/+vFlMqZWs8ZF44FfWRvN1hl1ZBfeugHAZccMd71K3gDc+zn9TepGSekJXEmbKA+6h1EuvS+8Rtjw7e9kpD3/xBwzQ4TRCvD789iXahsw2ijeFDtGXzzecuA0YrTVjysGRv/Hktpb1hY3qT9oc/w06izbudeqdlCGLQg/MhciNl4mzTihUUIZb4jvkAfABIYrX6bHzvbbwb3Dcd5P037iTBjlk/pi97pXk4VS5dgjQnC5jtg9hUijQOmqqrKVWY5i9xdE+PkrdRoLX6rCi1toPjMmBNf2N8b0gJgeIEVTD26zrgjWeIAFww0=);whiteSpace=wrap;html=1;labelPosition=center;verticalLabelPosition=bottom;align=center;verticalAlign=top;'

class DrawIOInterface:
    """
    Mixin class for objects that can be used with a drawio exporter.
    """

    @property
    def drawio_label(self) -> str:
        """
        The label of the object as a drawio compatible string.
        """
        raise NotImplementedError()

    @property
    def drawio_style(self) -> Dict[str, Any]:
        """
        The style of the object.
        """
        raise NotImplementedError()
