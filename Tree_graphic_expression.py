import ast
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ============================================================
# 1. Expresión del árbol
# ============================================================

expression = """
protected_div(sub(mul(protected_div(x210, x122), sub(x72, x19)), protected_div(mul(sub(x263, x314), protected_div(x236, x24)), x306)), sub(sub(add(x126, sub(x298, x236)), mul(x259, sub(x298, add(x141, x298)))), mul(x102, x136)))
"""


# ============================================================
# 2. Nodo del árbol
# ============================================================

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.x = 0
        self.y = 0


# ============================================================
# 3. Convertir la expresión a árbol
# ============================================================

def ast_to_tree(node):
    # Variables / terminales
    if isinstance(node, ast.Name):
        return Node(node.id)

    # Funciones
    if isinstance(node, ast.Call):

        if not isinstance(node.func, ast.Name):
            raise ValueError("Función no válida.")

        root = Node(node.func.id)

        for arg in node.args:
            root.children.append(ast_to_tree(arg))

        return root

    raise ValueError(
        f"Elemento no permitido en la expresión: {ast.dump(node)}"
    )


tree = ast_to_tree(
    ast.parse(expression, mode="eval").body
)


# ============================================================
# 4. Asignar posiciones a los nodos
# ============================================================

def calculate_positions(node, depth=0, x_counter=None):

    if x_counter is None:
        x_counter = [0]

    node.y = -depth

    # Si es hoja
    if not node.children:
        node.x = x_counter[0]
        x_counter[0] += 1

    else:
        # Primero posicionamos los hijos
        for child in node.children:
            calculate_positions(
                child,
                depth + 1,
                x_counter
            )

        # El padre queda centrado respecto a sus hijos
        node.x = (
            node.children[0].x +
            node.children[-1].x
        ) / 2

    return x_counter[0]


calculate_positions(tree)


# ============================================================
# 5. Obtener todos los nodos
# ============================================================

def get_nodes(node):

    nodes = [node]

    for child in node.children:
        nodes.extend(get_nodes(child))

    return nodes


nodes = get_nodes(tree)


# ============================================================
# 6. Dibujar el árbol
# ============================================================

fig, ax = plt.subplots(figsize=(18, 11))


# ------------------------------------------------------------
# Colores de los tipos de nodo
# ------------------------------------------------------------

operator_colors = {
    "protected_div": "#D9D2F3",
    "sub": "#F4CCCC",
    "mul": "#CFE2F3",
    "add": "#D9EAD3"
}

terminal_color = "#EDEDED"


# ------------------------------------------------------------
# Dibujar conexiones
# ------------------------------------------------------------

def draw_edges(node):

    for child in node.children:

        ax.plot(
            [node.x, child.x],
            [node.y, child.y],
            color="black",
            linewidth=1.2,
            zorder=1
        )

        draw_edges(child)


draw_edges(tree)


# ------------------------------------------------------------
# Dibujar nodos
# ------------------------------------------------------------

for node in nodes:

    if node.children:
        color = operator_colors.get(
            node.value,
            "#FFFFFF"
        )

        radius = 0.32

    else:
        color = terminal_color
        radius = 0.27

    circle = Circle(
        (node.x, node.y),
        radius=radius,
        facecolor=color,
        edgecolor="black",
        linewidth=1.2,
        zorder=2
    )

    ax.add_patch(circle)

    ax.text(
        node.x,
        node.y,
        node.value,
        ha="center",
        va="center",
        fontsize=9,
        zorder=3
    )


# ============================================================
# 7. Ajustes de la figura
# ============================================================

ax.set_aspect("equal")

ax.axis("off")

padding = 0.7

ax.set_xlim(
    min(n.x for n in nodes) - padding,
    max(n.x for n in nodes) + padding
)

ax.set_ylim(
    min(n.y for n in nodes) - padding,
    max(n.y for n in nodes) + padding
)


# ============================================================
# 8. Guardar como SVG
# ============================================================

# plt.savefig(
#     "tree_gp.svg",
#     format="svg",
#     bbox_inches="tight",
#     pad_inches=0.2
# )

plt.show()