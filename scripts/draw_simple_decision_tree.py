import graphviz

dot = graphviz.Digraph('simple-decision-tree', comment='Simple decision tree', format='png')

dot.node('A', 'Ima kljun?')
dot.node('B', 'Duljina vrata ≥ 1m?')
dot.edge('A','B','ne')
dot.node('C','Klokan')
dot.edge('B','C','ne')
dot.node('D','Žirafa')
dot.edge('B','D','da')

dot.node('E', 'Je li sisavac?')
dot.edge('A','E','da')
dot.node('F','Patka')
dot.edge('E','F','ne')
dot.node('G','Čudnovati kljunaš')
dot.edge('E','G','da')

dot.render(directory='img')