import fiona

def list_layers(gdb_path):
    """
    列出Geodatabase文件中的所有图层。

    参数:
    gdb_path (str): Geodatabase文件的路径。

    返回:
    list: 图层名称列表。
    """
    layers = fiona.listlayers(gdb_path)
    return layers

gdb_path = r"E:\桌面\实验\GreatLakes.gdb"  # 替换为你的Geodatabase文件路径
layers = list_layers(gdb_path)
print("Geodatabase文件中的图层有:", layers)
