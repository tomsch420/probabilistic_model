from N2G import drawio_diagram

from probabilistic_model.probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuit


class DrawIoExporter:

    model: ProbabilisticCircuit

    def __init__(self, model: ProbabilisticCircuit):
        self.model = model

    def export(self) -> drawio_diagram:
        diagram = drawio_diagram()
        diagram.add_diagram("Structure", width=1360, height=1864)
        for node in self.model.nodes:
            diagram.add_node(id=str(hash(node)), **node.draw_io_style())

        for source, target in self.model.unweighted_edges:
            diagram.add_link(str(hash(source)), str(hash(target)), style='endArrow=classic;html=1;rounded=0;')

        for source, target, weight in self.model.weighted_edges:
            diagram.add_link(str(hash(source)), str(hash(target)), label=f"{round(weight,2)}", style='endArrow=classic;html=1;rounded=0;')

        diagram.layout(algo="rt_circular")
        return diagram


"""
Edges: ->
Encoded : endArrow=classic;html=1;rounded=0;
Smooth:
<shape h="100" w="100" aspect="fixed" strokewidth="inherit">
  <connections>
    <constraint x="0" y="0" perimeter="1" />
    <constraint x="0.5" y="0" perimeter="1" />
    <constraint x="1" y="0" perimeter="1" />
    <constraint x="0" y="0.5" perimeter="1" />
    <constraint x="1" y="0.5" perimeter="1" />
    <constraint x="0" y="1" perimeter="1" />
    <constraint x="0.5" y="1" perimeter="1" />
    <constraint x="1" y="1" perimeter="1" />
  </connections>
  <background>    
  </background>
  <foreground>
    <fillstroke />
    <strokecolor color="#000000"/>
    <strokewidth width="2"/>
     <path>
      <move x="50" y="15"/>
      <line x="50" y="85"/>
      <move x="15" y="50"/>
      <line x="85" y="50"/>
     </path>
     <stroke/>
      <path>
      <move x="0" y="0"/>
    <ellipse x="0" y="0" w="100" h="100" />
     </path>
    <stroke />
  </foreground>
</shape>


encode: shape=stencil(tZVtb4MgEMc/DW8XHmLSt43bvgdTOkkpEKBr9+2LoGvxgXULGqPh/tzPu5MDQGrbUc0Ahh0grwBjBKF/+vFlMqZWs8ZF44FfWRvN1hl1ZBfeugHAZccMd71K3gDc+zn9TepGSekJXEmbKA+6h1EuvS+8Rtjw7e9kpD3/xBwzQ4TRCvD789iXahsw2ijeFDtGXzzecuA0YrTVjysGRv/Hktpb1hY3qT9oc/w06izbudeqdlCGLQg/MhciNl4mzTihUUIZb4jvkAfABIYrX6bHzvbbwb3Dcd5P037iTBjlk/pi97pXk4VS5dgjQnC5jtg9hUijQOmqqrKVWY5i9xdE+PkrdRoLX6rCi1toPjMmBNf2N8b0gJgeIEVTD26zrgjWeIAFww0=);whiteSpace=wrap;html=1;labelPosition=center;verticalLabelPosition=bottom;align=center;verticalAlign=top;

Determenistic: 
<shape h="100" w="100" aspect="fixed" strokewidth="inherit">
  <connections>
    <constraint x="0" y="0" perimeter="1" />
    <constraint x="0.5" y="0" perimeter="1" />
    <constraint x="1" y="0" perimeter="1" />
    <constraint x="0" y="0.5" perimeter="1" />
    <constraint x="1" y="0.5" perimeter="1" />
    <constraint x="0" y="1" perimeter="1" />
    <constraint x="0.5" y="1" perimeter="1" />
    <constraint x="1" y="1" perimeter="1" />
  </connections>
  <background>    
  </background>
  <foreground>
    <fillstroke />
    <strokecolor color="#000000"/>
    <strokewidth width="2"/>
     <path>
      <move x="50" y="15"/>
      <line x="50" y="85"/>
      <move x="15" y="50"/>
      <line x="85" y="50"/>
     </path>
     <stroke/>
      <path>
      <move x="0" y="0"/>
    <ellipse x="0" y="0" w="100" h="100" />
     </path>
    <stroke />
     <strokewidth width="1"/>
      <path>
      <move x="0" y="0"/>
      <ellipse x="7" y="7" w="86" h="86" />
     </path>
    <stroke />
  </foreground>
</shape>

encoded: shape=stencil(vZVtb4MgEMc/DW8XhLj6dnHb92B6naQUDLC1+/ZD0LX4QLtFZ4zm7uDn3d8DEC1Nw1pABDeIPiNCMozd09mnkc1MC5UNzj0/Qx3cxmp1gBOvbQ/gsgHNbRelLwg/uTHdTctKSekIXEkTRa7iDsa4dHPxOcD6b39FVuv4R7Cg+wyDF5HX+7EP+TbgbKN8Y+yQ/er5rgeOM862+nGrgbO/Y2npPEvNTcs3Vh3etfqQ9XTWYmyvNMwEfsJciLDwEmWGAZUSSjtHePs6EKHYX2mZrle22w4uK5yk57WsGzgJDOGj+oSL7vmoUfIUe0AILpcRxV2IOIss7qo8qcx8FsVvEP7nL+g0CL+WwrNbaLoyEIK35hZjfECMD5BNSl9syhtr/t8l20WMXSRZ8RgpNpirCuanTbYR7w0nvnd8Aw==);whiteSpace=wrap;html=1;labelPosition=center;verticalLabelPosition=bottom;align=center;verticalAlign=top;

Decompstion:
<shape h="100" w="100" aspect="fixed" strokewidth="inherit">
  <connections>
    <constraint x="0" y="0" perimeter="1" />
    <constraint x="0.5" y="0" perimeter="1" />
    <constraint x="1" y="0" perimeter="1" />
    <constraint x="0" y="0.5" perimeter="1" />
    <constraint x="1" y="0.5" perimeter="1" />
    <constraint x="0" y="1" perimeter="1" />
    <constraint x="0.5" y="1" perimeter="1" />
    <constraint x="1" y="1" perimeter="1" />
  </connections>
  <background>    
  </background>
  <foreground>
    <fillstroke />
    <strokecolor color="#000000"/>
    <strokewidth width="2"/>
     <path>
      <move x="25" y="25"/>
      <line x="75" y="75"/>
      <move x="75" y="25"/>
      <line x="25" y="75"/>
     </path>
     <stroke/>
      <path>
      <move x="0" y="0"/>
    <ellipse x="0" y="0" w="100" h="100" />
     </path>
    <stroke />
  </foreground>
</shape>

encoded: shape=stencil(tZXbboQgEIafhtsGIY3XjW3fg+psJcsCAbe7ffsiSFc8dduiMZqZYT5/BgcQrWzLNCCCW0SfESEFxu7p7MvEZlZD3QXngV+hCW7bGXWEC2+6AcBlC4Z3fZS+IPzkxvQ3rWolpSNwJW0SGcUdjHHpcvE1wIZvfyaWdvwTdGAGhcGLyOv92IfHfcDFTnpTbFSfXW8+cKq42GvhsoGLv2Np5TxrPzet3lh9fDfqLJt51mrsoAwsBL7DXIjQeBvTDANqJZRxjvD280CEYn9tl2nc2W47uHU42c7TrB84C8TwSX3Are4kXc9oLrMjQnA5QpQporwLkaoo/6+C/EaFX/yVOsXC56rw4ha6PTMQgmv7E2N6QEwPkKxT92mzrvDecIB5xxc=);whiteSpace=wrap;html=1;labelPosition=center;verticalLabelPosition=bottom;align=center;verticalAlign=top;

"""
