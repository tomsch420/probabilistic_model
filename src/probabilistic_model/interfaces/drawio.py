class DrawIOInterface:
    """
    Mixin class for objects that can be used with a drawio exporter.
    """

    @property
    def drawio_label(self) -> str:
        """
        The label of the object.
        """
        raise NotImplementedError()

    @property
    def drawio_style(self) -> str:
        """
        The style of the object.
        """
        raise NotImplementedError()