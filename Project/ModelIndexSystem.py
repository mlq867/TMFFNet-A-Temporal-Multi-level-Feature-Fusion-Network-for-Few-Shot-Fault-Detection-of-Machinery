# model_index.py — 简化版模型索引系统
# 仅包含：添加模型（程序调用）、查看、删除（交互式命令）
# 使用 JSON 存储模型信息

from typing import Dict, List
import tkinter as tk
from tkinter import messagebox, ttk
import json
import os


# index_file = r'C:\Users\86198\Desktop\少样本\Code\fewShotProject\Model\ModelRegistry\model_index.json'
index_file = r'C:\Users\86198\Desktop\少样本\Code\fewShotProject\Model\TRMFNet-2\model_index.json'

class ModelIndex:
    """支持任意参数动态存储的模型索引管理器"""

    def __init__(self, index_file: str = index_file):
        self.index_file = index_file
        self.models: List[Dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.index_file):
            with open(self.index_file, "r", encoding="utf-8") as f:
                self.models = json.load(f)
        else:
            self.models = []

    def _save(self):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.models, f, indent=4, ensure_ascii=False)

    # ------------------------
    # 添加模型（任意参数）
    # ------------------------
    def addModel(self, model_id: int, file_path: str, **kwargs):
        if model_id == -1:
            # 新增记录
            entry = {"id": len(self.models) + 1, "filePath": file_path}
            entry.update(kwargs)
            self.models.append(entry)
            self._save()
            print(f"=> 新增模型记录：ID={entry['id']} 路径={file_path}")
            return entry['id']
        else:
            # 更新已有记录
            found = False
            for m in self.models:
                if m["id"] == model_id:
                    m.update({"filePath": file_path})
                    m.update(kwargs)
                    found = True
                    self._save()
                    print(f"=> 更新模型记录：ID={model_id} 路径={file_path}")
                    return model_id
            # 如果没找到 id，则新增
            print(f"=> 指定 ID={model_id} 未找到，新增记录。")
            entry = {"id": len(self.models) + 1, "filePath": file_path}
            entry.update(kwargs)
            self.models.append(entry)
            self._save()
            return entry['id']

    # ------------------------
    # 列出模型（动态输出所有字段）
    # ------------------------
    def listModels(self):
        if not self.models:
            print("当前没有已登记的模型。")
            return
        print("=== 模型列表 ===")
        for m in self.models:
            # 按 key 排序输出
            line = " | ".join(f"{k}={m.get(k, '-')}" for k in sorted(m.keys()))
            print(line)

    # ------------------------
    # 删除模型
    # ------------------------
    def deleteModel(self, model_id: int, remove_file: bool = False):
        match_item = None
        for m in self.models:
            if m["id"] == model_id:
                match_item = m
                break
        if not match_item:
            print("没有找到该 ID.")
            return

        if remove_file:
            path = match_item.get("filePath", "")
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"文件已删除: {path}")
                except Exception as e:
                    print(f"删除文件失败: {e}")

        self.models = [m for m in self.models if m["id"] != model_id]
        # 重新分配 ID
        for i, m in enumerate(self.models, start=1):
            m["id"] = i
        self._save()
        print(f"ID={model_id} 的模型记录已删除。")


# ======================================================
# 交互控制台
# ======================================================
def console():
    index = ModelIndex()

    while True:
        print("""
======= 模型索引系统 =======
1. 查看所有模型
2. 删除一个模型
3. 退出
===========================
""")

        cmd = input("请输入指令编号: ")

        if cmd == "1":
            index.listModels()

        elif cmd == "2":
            mid = int(input("请输入要删除的模型 ID: "))
            index.deleteModel(mid)

        elif cmd == "3":
            print("退出系统。")
            break

        else:
            print("无效指令，请重新输入。")


# ======================
# Tkinter 增强模型管理 GUI（动态列 + 详情面板）
# ======================
class ModelIndexGUI:
    """
    优化后的模型管理界面：
    - 自动获取所有模型记录中的所有参数 key
    - 用户可选择哪些参数作为主表格列展示
    - 选中一条记录后，下方显示所有参数的详细信息
    """

    def __init__(self, index_file=index_file):
        self.index_file = index_file
        self._ensureFile()
        self.data = self._load()

        self.root = tk.Tk()
        self.root.title("模型索引管理器（动态列）")
        self.root.geometry("850x600")

        # -----------------------------
        # 1. 动态列选择区
        # -----------------------------
        self.columnFrame = tk.LabelFrame(self.root, text="选择要展示的列（参数）")
        self.columnFrame.pack(fill=tk.X, padx=5, pady=5)

        all_keys = self._collectAllKeys()
        self.columnVars = {}

        max_per_row = 6  # 每行最多多少个，可调

        for idx, key in enumerate(all_keys):
            var = tk.BooleanVar(value=key in ["id", "modelName", "dataset", "accuracy"])
            self.columnVars[key] = var

            # 自动换行：按 max_per_row 分行
            row = idx // max_per_row
            col = idx % max_per_row

            cb = tk.Checkbutton(self.columnFrame, text=key, variable=var,
                                command=self._rebuildTable)
            cb.grid(row=row, column=col, sticky="w", padx=5, pady=3)
        # for key in all_keys:
        #     var = tk.BooleanVar(value= key in ["id", "modelName", "dataset", "accuracy"])
        #     self.columnVars[key] = var
        #     tk.Checkbutton(self.columnFrame, text=key, variable=var, command=self._rebuildTable).pack(side=tk.LEFT)

        # -----------------------------
        # 2. 表格区
        # -----------------------------
        self.tableFrame = tk.Frame(self.root)
        self.tableFrame.pack(fill=tk.BOTH, expand=True)

        self.tree = None
        self._rebuildTable()

        # -----------------------------
        # 3. 详情显示区
        # -----------------------------
        self.detailFrame = tk.LabelFrame(self.root, text="模型详细信息")
        self.detailFrame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

        self.detailText = tk.Text(self.detailFrame, height=10)
        self.detailText.pack(fill=tk.BOTH)

        # -----------------------------
        # 4. 按钮区
        # -----------------------------
        btnFrame = tk.Frame(self.root)
        btnFrame.pack(fill=tk.X)
        tk.Button(btnFrame, text="刷新", command=self.refresh).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(btnFrame, text="删除选中", command=self.deleteSelected).pack(side=tk.LEFT, padx=5, pady=5)

        self.root.mainloop()

    # ==========================================================
    # 基础数据管理
    # ==========================================================
    def _ensureFile(self):
        if not os.path.exists(self.index_file):
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _load(self):
        with open(self.index_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _collectAllKeys(self):
        keys = set()
        for item in self.data:
            keys.update(item.keys())
        return sorted(list(keys))

    # ==========================================================
    # 动态重建表格
    # ==========================================================
    def _rebuildTable(self):
        if self.tree:
            self.tree.destroy()

        selected_columns = [k for k, v in self.columnVars.items() if v.get()]
        if not selected_columns:
            selected_columns = ["id", "modelName"]

        # ---- 固定 id 为第一列 ----
        if "id" in selected_columns:
            selected_columns = ["id"] + [col for col in selected_columns if col != "id"]

        self.tree = ttk.Treeview(self.tableFrame, columns=selected_columns, show="headings")
        for col in selected_columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.tree.bind("<<TreeviewSelect>>", self._showDetails)
        self._fillTable()

    # ==========================================================
    # 填充表格内容
    # ==========================================================
    def _fillTable(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        selected_columns = self.tree["columns"]

        for item in self.data:
            row = []
            for col in selected_columns:
                row.append(item.get(col, "-"))
            self.tree.insert("", tk.END, values=row)

    # ==========================================================
    # 显示详情
    # ==========================================================
    def _showDetails(self, event):
        selected = self.tree.selection()
        if not selected:
            return

        values = self.tree.item(selected[0], "values")
        columns = self.tree["columns"]

        # 找到对应字典
        match_item = None
        for item in self.data:
            ok = True
            for col, val in zip(columns, values):
                if str(item.get(col)) != str(val):
                    ok = False
                    break
            if ok:
                match_item = item
                break

        if match_item:
            self.detailText.delete("1.0", tk.END)
            for k,v in match_item.items():
                self.detailText.insert(tk.END, f"{k}: {v}\n")

    # ==========================================================
    # 删除记录
    # ==========================================================
    def deleteSelected(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("提示", "未选中记录")
            return

        values = self.tree.item(selected[0], "values")
        columns = self.tree["columns"]

        match_item = None
        for item in self.data:
            ok = True
            for col, val in zip(columns, values):
                if str(item.get(col)) != str(val):
                    ok = False
                    break
            if ok:
                match_item = item
                break

        if not match_item:
            return

        file_path = match_item.get("filePath", "")

        if messagebox.askyesno("确认", f"删除此记录并删除模型文件？\n{file_path}"):
            # 删除文件
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    messagebox.showerror("错误", f"文件删除失败：{e}")

            # 删除记录
            self.data = [x for x in self.data if x != match_item]
            self._save(self.data)
            self.refresh()
            messagebox.showinfo("成功", "删除完成！")

    # ==========================================================
    # 刷新
    # ==========================================================
    def refresh(self):
        self.data = self._load()
        # --------- 重新分配 ID（从 1 开始连续）---------
        for i, item in enumerate(self.data, start=1):
            item["id"] = i
        # 保存回文件
        self._save(self.data)
        self._rebuildTable()

    def _ensure_file(self):
        if not os.path.exists(self.index_file):
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def load(self):
        with open(self.index_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, data):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def loadModels(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        data = self.load()
        for item in data:
            self.tree.insert("", tk.END, values=(
                item.get("id"), item.get("modelName"), item.get("dataset"), item.get("n"),
                item.get("k"), item.get("accuracy"), item.get("filePath")
            ))


# GUI 入口
if __name__ == "__main__":
    ModelIndexGUI()
