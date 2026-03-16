"""
local_zep 本体存根
与 zep_cloud.external_clients.ontology 接口兼容

graph_builder.py 使用 type(name, (EntityModel,), attrs) 动态创建子类，
这些类只需要存在，不需要业务逻辑。
"""


class EntityModel:
    """实体基类（存根）"""
    pass


class EntityText:
    """实体文本字段类型（存根）"""
    pass


class EdgeModel:
    """边基类（存根）"""
    pass
