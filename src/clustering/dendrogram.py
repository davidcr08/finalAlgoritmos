"""Visualización de dendrogramas con matplotlib (sin cruces)."""
import matplotlib
matplotlib.use("Agg")  # Backend no interactivo
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Literal

from src.clustering.hierarchical import LinkageMatrix


def _dendrogram_coords(linkage: LinkageMatrix) -> tuple[dict, dict]:
    """
    Calcula coordenadas (x, y) para dibujar dendrograma sin cruces.
    Algoritmo recursivo: posiciona hojas en orden, clusters internos
    centrados sobre sus hijos.
    
    Returns:
        leaf_positions: {leaf_id: (x, y)} - y siempre 0 para hojas
        internal_positions: {cluster_id: (x, y)} - y = height
    """
    n = linkage.n_leaves
    matrix = linkage.matrix
    
    # Árbol: cluster_id -> {left, right, height, leaves_list}
    tree = {}
    for i in range(n):
        tree[i] = {"left": None, "right": None, "height": 0.0, "leaves": [i]}
    
    for row_idx, row in enumerate(matrix):
        id_a, id_b, height, _ = row
        id_a, id_b = int(id_a), int(id_b)
        new_id = n + row_idx
        leaves_a = tree[id_a]["leaves"]
        leaves_b = tree[id_b]["leaves"]
        tree[new_id] = {
            "left": id_a,
            "right": id_b,
            "height": height,
            "leaves": leaves_a + leaves_b,
        }
    
    # Posiciones de hojas: ordenadas por aparición en el último cluster (orden natural)
    root_id = n + len(matrix) - 1
    leaf_order = tree[root_id]["leaves"]
    
    # Asignar x equidistantes a hojas
    leaf_positions = {}
    for idx, leaf_id in enumerate(leaf_order):
        leaf_positions[leaf_id] = (float(idx), 0.0)
    
    # Posiciones internas: x = promedio de hijos, y = height
    internal_positions = {}
    for row_idx in range(len(matrix)):
        row = matrix[row_idx]
        id_a, id_b, height, _ = row
        id_a, id_b = int(id_a), int(id_b)
        new_id = n + row_idx
        
        x_a = leaf_positions[id_a][0] if id_a < n else internal_positions[id_a][0]
        x_b = leaf_positions[id_b][0] if id_b < n else internal_positions[id_b][0]
        
        internal_positions[new_id] = ((x_a + x_b) / 2.0, height)
    
    return leaf_positions, internal_positions


def plot_dendrogram(
    linkage: LinkageMatrix,
    labels: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    Dibuja dendrograma a partir de linkage matrix.
    
    Args:
        linkage: LinkageMatrix con n-1 filas [id_a, id_b, height, size]
        labels: Etiquetas para hojas (ej: ["A01", "A02", ...]). Si None, usa índices.
        title: Título del gráfico
        figsize: Tamaño de figura
        save_path: Ruta para guardar (PNG). Si None, no guarda.
        show: Si True, muestra en pantalla (requiere backend interactivo)
    
    Returns:
        matplotlib Figure
    """
    n = linkage.n_leaves
    matrix = linkage.matrix
    
    if labels is None:
        labels = [str(i) for i in range(n)]
    assert len(labels) == n
    
    leaf_pos, internal_pos = _dendrogram_coords(linkage)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Dibujar líneas verticales y horizontales
    for row_idx, row in enumerate(matrix):
        id_a, id_b, height, _ = row
        id_a, id_b = int(id_a), int(id_b)
        new_id = n + row_idx
        
        # Coordenadas
        x_a = leaf_pos[id_a][0] if id_a < n else internal_pos[id_a][0]
        x_b = leaf_pos[id_b][0] if id_b < n else internal_pos[id_b][0]
        y_a = leaf_pos[id_a][1] if id_a < n else internal_pos[id_a][1]
        y_b = leaf_pos[id_b][1] if id_b < n else internal_pos[id_b][1]
        y_new = height
        
        # Líneas verticales desde hijos hasta altura de fusión
        ax.plot([x_a, x_a], [y_a, y_new], "k-", linewidth=1)
        ax.plot([x_b, x_b], [y_b, y_new], "k-", linewidth=1)
        
        # Línea horizontal conectando las dos verticales
        ax.plot([x_a, x_b], [y_new, y_new], "k-", linewidth=1)
    
    # Etiquetas de hojas
    for leaf_id in range(n):
        x, _ = leaf_pos[leaf_id]
        ax.text(x, -0.01, labels[leaf_id], ha="center", va="top", fontsize=9)
    
    # Configuración ejes
    ax.set_ylabel("Distancia de enlace")
    if title:
        ax.set_title(title)
    ax.set_xlim(-0.5, n - 0.5)
    
    # Y límite: max height + margen
    max_height = np.max(matrix[:, 2]) if len(matrix) > 0 else 1.0
    ax.set_ylim(-0.05, max_height * 1.1)
    
    # Invertir Y para que 0 esté abajo (convención dendrograma)
    ax.invert_yaxis()
    
    # Quitar ticks X
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_all_dendrograms(
    linkages: dict[str, LinkageMatrix],
    labels: list[str],
    output_dir: str | Path,
    prefix: str = "dendrogram",
) -> dict[str, Path]:
    """
    Genera los 3 dendrogramas (single, complete, average) y los guarda.
    
    Returns:
        dict: {method: Path}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    for method, linkage in linkages.items():
        title = f"Dendrograma - {method.capitalize()} Linkage"
        save_path = output_dir / f"{prefix}_{method}.png"
        plot_dendrogram(linkage, labels, title, save_path=save_path)
        paths[method] = save_path
    
    return paths


def _demo():
    """Demo rápido."""
    import numpy as np
    from src.clustering.hierarchical import agglomerative_clustering
    
    np.random.seed(42)
    n = 10
    X = np.random.rand(n, 5)
    D = 1 - np.corrcoef(X)
    D = np.clip(D, 0, 1)
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    
    labels = [f"A{i+1:02d}" for i in range(n)]
    
    linkages = {}
    for method in ["single", "complete", "average"]:
        linkages[method] = agglomerative_clustering(D, method)
    
    plot_all_dendrograms(linkages, labels, "data/processed/plots", "demo")
    print("Dendrogramas guardados en data/processed/plots/")


if __name__ == "__main__":
    _demo()