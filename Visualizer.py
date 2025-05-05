import graphviz

def visualize_tree(tree, graph=None, node_id=0):
       if graph is None:
         graph = graphviz.Digraph(comment='Decision Tree', graph_attr={'rankdir': 'TB'})
       if type(tree)==str:
         graph.node(str(node_id), label=f"Class: {tree}")
       else:
         #print(tree)
         graph.node(str(node_id), label=f"Feature: {tree['Label']}\nThreshold: {tree['val']}\n Gini:{tree['gini']}")
         left_child_id = node_id * 2 + 1
         right_child_id = node_id * 2 + 2
         graph.edge(str(node_id), str(left_child_id), label='Left')
         graph.edge(str(node_id), str(right_child_id), label='Right')
         visualize_tree(tree['Left'], graph, left_child_id)
         visualize_tree(tree['Right'], graph, right_child_id)
       return graph
if __name__=='__main__':
    tree = {
       'Label': 'feature1', 'val': 0.5,
       'Left': {
           'Label': 'feature2', 'val': 0.3,
           'Left': "yes",
           'Right': "no"
       },
       'Right': "yes"
   }
    dot = visualize_tree(tree)
    dot.render('decision_tree', view=True, format='png')
