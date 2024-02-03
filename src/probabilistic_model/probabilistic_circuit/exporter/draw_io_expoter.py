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

# Edges Tagret von Source nach Target
# DrawIO_style return Dict für Style
# Draw IO + x und OO +
# Smooth ⊕  ⊗
#shape=mxgraph.cisco.misc.asr_1000_series;html=1;pointerEvents=1;dashed=0;fillColor=#036897;strokeColor=#ffffff;strokeWidth=2;verticalLabelPosition=bottom;verticalAlign=top;align=center;outlineConnect=0;
# Decompo Product ⊕ * 45°
#shape=mxgraph.cisco.misc.asr_1000_series;html=1;pointerEvents=1;dashed=0;fillColor=#036897;strokeColor=#ffffff;strokeWidth=2;verticalLabelPosition=bottom;verticalAlign=top;align=center;outlineConnect=0;
# Detemisting SUm o⊕
#shape=mxgraph.cisco.misc.asr_1000_series;html=1;pointerEvents=1;dashed=0;fillColor=#036897;strokeColor=#ffffff;strokeWidth=2;verticalLabelPosition=bottom;verticalAlign=top;align=center;outlineConnect=0;
# repsentation von distri ampassen wie nötig uniform etc.
# Alog für net kaka
# Leaf mit Var namen
#



"""
Edges: ->
Encoded : endArrow=classic;html=1;rounded=0;
Decompostion:
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
    <ellipse x="0" y="0" w="100" h="100" />
  </background>
  <foreground>
    <fillstroke />
     <path>
     <strokewidth width="2"/>
      <move x="20" y="25"/>
      <line x="80" y="75"/>
      <move x="80" y="25"/>
      <line x="20" y="75"/>
     </path>
    <stroke />
  </foreground>
</shape>


encode: shape=stencil(tZRvb4MgEMY/DW8bhDTr24Vu34PqWUkpEKB/9u2LopnY6tymiSG55+THkzsORJmruAFEcIXoHhGSYRzWEN8GMXcGch/FUtyhiLLzVp/gJgrfAoSqwApfZ+kHwu/hn/qjLNdKBYLQyiWZXj7AuFBhL75HWHv2VxKZwD+DB9s6jCoin/Oxm+064Gwlvym2c7+43+XAqeNsrcYtBs7+jqUsKGOXm7IDz09Hqy+qeGkGpBTGwQ9XfjiMw2F96WnkZMpKbWHCUimkjGM9UUTD64F/SnTp/rMQ7H8/D2SqN93us772KjIoCdnOQUiheohdinibhUhd7P7vgvzGRdPCkSp39R3r/FODGzW+9I3wAA==);whiteSpace=wrap;html=1;labelPosition=center;verticalLabelPosition=bottom;align=center;verticalAlign=top;

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
    <ellipse x="0" y="0" w="100" h="100" />
  </background>
  <foreground>
    <save/>
    <fillstroke />
    <strokewidth width="2"/>
     <path>
      <move x="50" y="15"/>
      <line x="50" y="85"/>
      <move x="15" y="50"/>
      <line x="85" y="50"/>
     </path>
     <stroke/>
     <strokewidth width="1"/>
      <restore/>
      <path>
      <move x="0" y="0"/>
      <ellipse x="5" y="5" w="90" h="90" />
     </path>
    <stroke />
  </foreground>
</shape>

encoded: shape=stencil(tZXdboQgEIWfhttGISa7l41t34PqWMmyYID96dsXRVNBZXeNJsaEGfw4mZmDiOS6pg0gnNSIfCCM0ySxb7u+BWuqGyiMC1bsDqULa6PkCW6sND2AiRoUM22WfKLk3e5pH5IXUghLYFJoLzPKWxhlwn6b3B2sP/vXWzWWfwYDqlfoogh/PY99y/YBpzvp9bGD+s31bgf2Fad7NW4zcLoeS3IbWRpukn/T4vSj5EWUs2KAc9ZoeDDyoRlDs85qWjiZ5JVUEJGk6RUi5asY5871kU3ja8HK/78ecLw3DW03ThJD+iyvo1JlwZRlMfaA4EwsIw5PIXwVqT+SA/EVFYdXEF1vF+o0FH5NZx64RoE2dm626t3smMdrNrFKUDXPKkffKcfoCStL2n02MVMXdX/VLvAH);whiteSpace=wrap;html=1;labelPosition=center;verticalLabelPosition=bottom;align=center;verticalAlign=top;
smooth:
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
    <ellipse x="0" y="0" w="100" h="100" />
  </background>
  <foreground>
    <fillstroke />
     <path>
     <strokewidth width="2"/>
      <move x="50" y="15"/>
      <line x="50" y="85"/>
      <move x="15" y="50"/>
      <line x="85" y="50"/>
     </path>
    <stroke />
  </foreground>
</shape>

encoded: shape=stencil(tZTtboMgFIavhr8LHzHp34Zt98EUKykFAmzt7n4okoktzi6aGJJzjjy8OV+AUNcxwwGGHSCvAGMEYTiDfZ3ZzBle++hsxY030e281Wd+FY0fAUJ13ArfR8kbgMfwT/8RWmulAkFo5bLIJB5gTKhwF94ibHz7O7NM4F+453ZUGL0Av6/HvlT7gNFOenNsUr+53u3AuWK0V+E2A6P/YwkNnlJzE/rB6vPJ6k/VPBTDpRTG8T9afj6M82F9qKnwMqGttnxBUiukjGO9kETD+oG/C6TwdC0E+b/rAS/VJt2+6K9JRqpZM1VrEFKoMuKwCpGrQHnnJeIzKg7PIIYSFrKc8luq/F2BB2/c9IPjBw==);whiteSpace=wrap;html=1;labelPosition=center;verticalLabelPosition=bottom;align=center;verticalAlign=top;

"""

