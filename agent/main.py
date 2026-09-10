#!/usr/bin/env python3
"""
code_agent.py - A local coding agent GUI built with PySide6.
Single-file implementation.
"""

import sys
import os
import re
import json
import shutil
import difflib
import subprocess
import threading
import traceback
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable, Tuple

from PySide6.QtCore import (
    Qt,
    QThread,
    Signal,
    QProcess,
    QModelIndex,
    QObject,
    QDir,
    QSize,
    QProcessEnvironment,
    QSettings,
    QPoint,
    QRect,
    QStandardPaths,
    QTimer,
    QUrl,
    QEvent,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QFontDatabase,
    QKeySequence,
    QPainter,
    QTextCharFormat,
    QTextCursor,
    QTextFormat,
    QSyntaxHighlighter,
    QIcon,
    QDesktopServices,
    QCloseEvent,
    QContextMenuEvent,
    QTextDocument,
    QPalette,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QTreeView,
    QTabWidget,
    QPlainTextEdit,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QLabel,
    QToolBar,
    QStatusBar,
    QFileDialog,
    QMessageBox,
    QInputDialog,
    QMenu,
    QDialog,
    QFormLayout,
    QDialogButtonBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QTextBrowser,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyleFactory,
    QFileSystemModel,
)

from agent import brain as agent_brain
from agent import tools as agent_tools
from agent.brain import ask_model

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
APP_NAME = "Code Agent"
ORG_NAME = "LocalTools"
SETTINGS_FILE = "settings.ini"
BACKUP_DIR_NAME = ".code_agent_backup"
HIDDEN_DIRS = {".git", "__pycache__", ".venv", "node_modules", BACKUP_DIR_NAME}


# ----------------------------------------------------------------------
# Syntax Highlighter
# ----------------------------------------------------------------------
class CodeHighlighter(QSyntaxHighlighter):
    """Basic syntax highlighter supporting multiple languages."""

    def __init__(self, document: QTextDocument, language: str = "plain"):
        super().__init__(document)
        self.language = language
        self.rules = []
        self.setup_rules()

    def setup_rules(self):
        # Define formats
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#569CD6"))
        keyword_format.setFontWeight(QFont.Bold)

        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#CE9178"))

        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#6A9955"))
        comment_format.setFontItalic(True)

        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#B5CEA8"))

        function_format = QTextCharFormat()
        function_format.setForeground(QColor("#DCDCAA"))

        class_format = QTextCharFormat()
        class_format.setForeground(QColor("#4EC9B0"))

        self.rules = []

        # Keywords per language
        keywords = {
            "python": [
                "and",
                "as",
                "assert",
                "async",
                "await",
                "break",
                "class",
                "continue",
                "def",
                "del",
                "elif",
                "else",
                "except",
                "False",
                "finally",
                "for",
                "from",
                "global",
                "if",
                "import",
                "in",
                "is",
                "lambda",
                "None",
                "nonlocal",
                "not",
                "or",
                "pass",
                "raise",
                "return",
                "True",
                "try",
                "while",
                "with",
                "yield",
            ],
            "javascript": [
                "async",
                "await",
                "break",
                "case",
                "catch",
                "class",
                "const",
                "continue",
                "debugger",
                "default",
                "delete",
                "do",
                "else",
                "export",
                "extends",
                "finally",
                "for",
                "function",
                "if",
                "import",
                "in",
                "instanceof",
                "let",
                "new",
                "of",
                "return",
                "static",
                "super",
                "switch",
                "this",
                "throw",
                "try",
                "typeof",
                "var",
                "void",
                "while",
                "with",
                "yield",
            ],
            "typescript": [
                "async",
                "await",
                "break",
                "case",
                "catch",
                "class",
                "const",
                "continue",
                "debugger",
                "default",
                "delete",
                "do",
                "else",
                "export",
                "extends",
                "finally",
                "for",
                "function",
                "if",
                "import",
                "in",
                "instanceof",
                "interface",
                "let",
                "new",
                "of",
                "return",
                "static",
                "super",
                "switch",
                "this",
                "throw",
                "try",
                "typeof",
                "var",
                "void",
                "while",
                "with",
                "yield",
                "type",
                "namespace",
                "enum",
                "implements",
                "public",
                "private",
                "protected",
                "readonly",
            ],
            "java": [
                "abstract",
                "assert",
                "boolean",
                "break",
                "byte",
                "case",
                "catch",
                "char",
                "class",
                "const",
                "continue",
                "default",
                "do",
                "double",
                "else",
                "enum",
                "extends",
                "final",
                "finally",
                "float",
                "for",
                "goto",
                "if",
                "implements",
                "import",
                "instanceof",
                "int",
                "interface",
                "long",
                "native",
                "new",
                "package",
                "private",
                "protected",
                "public",
                "return",
                "short",
                "static",
                "strictfp",
                "super",
                "switch",
                "synchronized",
                "this",
                "throw",
                "throws",
                "transient",
                "try",
                "void",
                "volatile",
                "while",
            ],
            "c": [
                "auto",
                "break",
                "case",
                "char",
                "const",
                "continue",
                "default",
                "do",
                "double",
                "else",
                "enum",
                "extern",
                "float",
                "for",
                "goto",
                "if",
                "int",
                "long",
                "register",
                "return",
                "short",
                "signed",
                "sizeof",
                "static",
                "struct",
                "switch",
                "typedef",
                "union",
                "unsigned",
                "void",
                "volatile",
                "while",
            ],
            "cpp": [
                "auto",
                "break",
                "case",
                "char",
                "const",
                "continue",
                "default",
                "do",
                "double",
                "else",
                "enum",
                "extern",
                "float",
                "for",
                "goto",
                "if",
                "int",
                "long",
                "register",
                "return",
                "short",
                "signed",
                "sizeof",
                "static",
                "struct",
                "switch",
                "typedef",
                "union",
                "unsigned",
                "void",
                "volatile",
                "while",
                "class",
                "namespace",
                "template",
                "typename",
                "public",
                "private",
                "protected",
                "new",
                "delete",
                "this",
                "virtual",
                "override",
                "final",
                "using",
            ],
            "bash": [
                "if",
                "then",
                "else",
                "elif",
                "fi",
                "for",
                "while",
                "until",
                "do",
                "done",
                "case",
                "esac",
                "function",
                "select",
                "time",
                "coproc",
                "in",
                "return",
                "break",
                "continue",
            ],
            "sql": [
                "select",
                "from",
                "where",
                "insert",
                "update",
                "delete",
                "create",
                "drop",
                "alter",
                "table",
                "index",
                "view",
                "and",
                "or",
                "not",
                "null",
                "primary",
                "key",
                "foreign",
                "references",
                "join",
                "inner",
                "left",
                "right",
                "outer",
                "group",
                "by",
                "order",
                "having",
                "distinct",
                "limit",
                "offset",
            ],
            "yaml": [],
            "json": [],
            "html": [],
            "css": [],
            "markdown": [],
            "plain": [],
        }

        if self.language in keywords:
            for word in keywords[self.language]:
                pattern = r"\b" + re.escape(word) + r"\b"
                self.rules.append((re.compile(pattern), keyword_format))

        # Strings
        if self.language in [
            "python",
            "javascript",
            "typescript",
            "java",
            "c",
            "cpp",
            "bash",
            "sql",
        ]:
            string_pattern = re.compile(r"(\".*?\"|\'.*?\')")
            self.rules.append((string_pattern, string_format))

        # Comments
        if self.language in ["python", "bash", "yaml"]:
            comment_pattern = re.compile(r"#[^\n]*")
        elif self.language in ["javascript", "typescript", "java", "c", "cpp", "css"]:
            comment_pattern = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
        elif self.language in ["sql"]:
            comment_pattern = re.compile(r"--[^\n]*")
        elif self.language in ["html", "markdown"]:
            comment_pattern = re.compile(r"<!--.*?-->", re.DOTALL)
        else:
            comment_pattern = None
        if comment_pattern:
            self.rules.append((comment_pattern, comment_format))

        # Numbers
        if self.language in [
            "python",
            "javascript",
            "typescript",
            "java",
            "c",
            "cpp",
            "sql",
        ]:
            number_pattern = re.compile(r"\b\d+(\.\d+)?\b")
            self.rules.append((number_pattern, number_format))

        # Functions (simple heuristic)
        if self.language in ["python", "javascript", "typescript", "java", "c", "cpp"]:
            func_pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
            self.rules.append((func_pattern, function_format))

    def highlightBlock(self, text: str):
        for pattern, fmt in self.rules:
            for match in pattern.finditer(text):
                start, end = match.span()
                self.setFormat(start, end - start, fmt)

        # Multi-line strings for Python (simplified)
        if self.language == "python":
            self.setCurrentBlockState(0)
            # TODO: handle triple quotes (omitted for brevity)

    def set_language(self, language: str):
        self.language = language
        self.setup_rules()
        self.rehighlight()


# ----------------------------------------------------------------------
# Line Number Area
# ----------------------------------------------------------------------
class LineNumberArea(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.code_editor = editor

    def sizeHint(self):
        return QSize(self.code_editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.code_editor.line_number_area_paint_event(event)


# ----------------------------------------------------------------------
# Code Editor Widget
# ----------------------------------------------------------------------
class CodeEditor(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))

        self.line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.cursorPositionChanged.connect(self.highlight_current_line)

        self.highlighter = None
        self.language = "plain"
        self.file_path = None
        self.is_modified = False

        self.update_line_number_area_width(0)
        self.highlight_current_line()

        # Set default font
        font = QFont("Monospace")
        font.setStyleHint(QFont.TypeWriter)
        font.setPointSize(10)
        self.setFont(font)

    def line_number_area_width(self):
        digits = len(str(max(1, self.blockCount())))
        space = 3 + self.fontMetrics().horizontalAdvance("9") * digits
        return space

    def update_line_number_area_width(self, _):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect, dy):
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(
                0, rect.y(), self.line_number_area.width(), rect.height()
            )
        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(
            QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height())
        )

    def line_number_area_paint_event(self, event):
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor("#1E1E1E"))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = round(
            self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        )
        bottom = top + round(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QColor("#858585"))
                painter.drawText(
                    0,
                    top,
                    self.line_number_area.width() - 5,
                    self.fontMetrics().height(),
                    Qt.AlignRight,
                    number,
                )
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            block_number += 1

    def highlight_current_line(self):
        extra_selections = []
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            line_color = QColor("#2A2A2A")
            selection.format.setBackground(line_color)
            selection.format.setProperty(QTextFormat.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            extra_selections.append(selection)
        self.setExtraSelections(extra_selections)

    def set_language(self, language: str):
        self.language = language
        if not self.highlighter:
            self.highlighter = CodeHighlighter(self.document(), language)
        else:
            self.highlighter.set_language(language)

    def load_file(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.setPlainText(content)
            self.file_path = file_path
            self.is_modified = False
            ext = os.path.splitext(file_path)[1].lower()
            lang = self.detect_language(ext)
            self.set_language(lang)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load file: {e}")

    def save_file(self):
        if self.file_path:
            try:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    f.write(self.toPlainText())
                self.is_modified = False
                self.document().setModified(False)
                return True
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not save file: {e}")
                return False
        return False

    def detect_language(self, ext: str) -> str:
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".json": "json",
            ".html": "html",
            ".css": "css",
            ".md": "markdown",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".sh": "bash",
            ".bash": "bash",
            ".sql": "sql",
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".hpp": "cpp",
            ".java": "java",
            ".txt": "plain",
        }
        return ext_map.get(ext, "plain")

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.highlight_current_line()


# ----------------------------------------------------------------------
# Editor Tabs
# ----------------------------------------------------------------------
class EditorTabs(QTabWidget):
    file_saved = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(True)
        self.tabCloseRequested.connect(self.close_tab)
        self.currentChanged.connect(self.on_tab_changed)
        self.editors = []  # list of CodeEditor
        self.setDocumentMode(True)

    def new_editor(self, file_path: Optional[str] = None) -> CodeEditor:
        editor = CodeEditor()
        if file_path:
            editor.load_file(file_path)
            self.addTab(editor, os.path.basename(file_path))
            editor.file_path = file_path
        else:
            self.addTab(editor, "Untitled")
        self.editors.append(editor)
        self.setCurrentWidget(editor)
        editor.textChanged.connect(lambda: self.on_text_changed(editor))
        return editor

    def open_file(self, file_path: str):
        # Check if already open
        for i, editor in enumerate(self.editors):
            if editor.file_path == file_path:
                self.setCurrentIndex(i)
                return editor
        editor = self.new_editor(file_path)
        return editor

    def close_tab(self, index: int):
        editor = self.editors[index]
        if editor.is_modified:
            ret = QMessageBox.question(
                self,
                "Unsaved Changes",
                f"Save changes to {editor.file_path or 'Untitled'}?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if ret == QMessageBox.Save:
                if not editor.save_file():
                    return
            elif ret == QMessageBox.Cancel:
                return
        self.removeTab(index)
        self.editors.pop(index)
        editor.deleteLater()
        if len(self.editors) == 0:
            # Add a blank editor
            self.new_editor()

    def current_editor(self) -> Optional[CodeEditor]:
        if self.count() == 0:
            return None
        return self.currentWidget()

    def save_current(self) -> bool:
        editor = self.current_editor()
        if editor:
            return editor.save_file()
        return False

    def save_all(self):
        for editor in self.editors:
            if editor.is_modified:
                editor.save_file()

    def on_text_changed(self, editor: CodeEditor):
        editor.is_modified = True
        editor.document().setModified(True)
        idx = self.indexOf(editor)
        if idx >= 0:
            text = self.tabText(idx)
            if not text.endswith(" *"):
                self.setTabText(idx, text + " *")

    def on_tab_changed(self, index: int):
        if index >= 0 and index < len(self.editors):
            editor = self.editors[index]
            self.setWindowModified(editor.is_modified)

    def closeEvent(self, event):
        # Handle close with unsaved changes
        unsaved = [e for e in self.editors if e.is_modified]
        if unsaved:
            ret = QMessageBox.question(
                self,
                "Unsaved Changes",
                f"You have {len(unsaved)} unsaved file(s). Save all?",
                QMessageBox.SaveAll | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if ret == QMessageBox.SaveAll:
                self.save_all()
                event.accept()
            elif ret == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ----------------------------------------------------------------------
# Project Explorer
# ----------------------------------------------------------------------
class ProjectExplorer(QWidget):
    open_file_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_root = None
        self.model = QFileSystemModel()
        self.model.setFilter(
            QDir.AllEntries | QDir.NoDotAndDotDot | QDir.AllDirs | QDir.Files
        )
        self.model.setNameFilters([])
        self.model.setNameFilterDisables(False)

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setAnimated(True)
        self.tree.setIndentation(20)
        self.tree.setSortingEnabled(True)
        self.tree.setSelectionMode(QTreeView.SingleSelection)
        self.tree.setSelectionBehavior(QTreeView.SelectRows)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(False)
        self.tree.setExpandsOnDoubleClick(True)
        self.tree.setMinimumWidth(230)
        self.tree.setColumnWidth(0, 230)
        self.tree.setColumnHidden(1, True)
        self.tree.setColumnHidden(2, True)
        self.tree.setColumnHidden(3, True)
        self.tree.doubleClicked.connect(self.on_double_click)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)
        self.setLayout(layout)

        self.empty_label = QLabel("Open a project to browse files")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color: #858585; padding: 24px;")
        layout.addWidget(self.empty_label)
        self.tree.hide()

        self.refresh_timer = QTimer()
        self.refresh_timer.setInterval(3000)
        self.refresh_timer.timeout.connect(self.refresh)

    def set_project(self, root_path: str):
        self.project_root = root_path
        if not root_path:
            self.tree.hide()
            self.empty_label.show()
            self.refresh_timer.stop()
            return

        self.empty_label.hide()
        self.tree.show()
        self.model.setRootPath(root_path)
        self.tree.setRootIndex(self.model.index(root_path))
        # Hide hidden directories
        self.refresh()
        self.refresh_timer.start()

    def refresh(self):
        if self.project_root:
            self.model.setRootPath(self.project_root)
            self.tree.setRootIndex(self.model.index(self.project_root))
            self.tree.setColumnWidth(0, max(230, self.tree.width() - 12))

    def on_double_click(self, index: QModelIndex):
        if self.model.isDir(index):
            return
        file_path = self.model.filePath(index)
        if os.path.isfile(file_path):
            self.open_file_requested.emit(file_path)

    def show_context_menu(self, pos):
        index = self.tree.indexAt(pos)
        menu = QMenu()
        if index.isValid():
            file_path = self.model.filePath(index)
            if self.model.isDir(index):
                menu.addAction("New File", lambda: self.create_file(file_path))
                menu.addAction("New Folder", lambda: self.create_folder(file_path))
                menu.addSeparator()
                menu.addAction("Rename", lambda: self.rename_entry(file_path))
                menu.addAction("Delete", lambda: self.delete_entry(file_path))
                menu.addSeparator()
                menu.addAction(
                    "Open in Terminal", lambda: self.open_in_terminal(file_path)
                )
            else:
                menu.addAction("Open", lambda: self.open_file_requested.emit(file_path))
                menu.addSeparator()
                menu.addAction("Rename", lambda: self.rename_entry(file_path))
                menu.addAction("Delete", lambda: self.delete_entry(file_path))
                menu.addSeparator()
                menu.addAction(
                    "Open Containing Folder",
                    lambda: self.open_containing_folder(file_path),
                )
        else:
            menu.addAction("New File", lambda: self.create_file(self.project_root))
            menu.addAction("New Folder", lambda: self.create_folder(self.project_root))
            menu.addSeparator()
            menu.addAction("Refresh", self.refresh)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def create_file(self, parent_dir: str):
        name, ok = QInputDialog.getText(self, "New File", "File name:")
        if ok and name:
            file_path = os.path.join(parent_dir, name)
            if not os.path.exists(file_path):
                try:
                    open(file_path, "w").close()
                    self.refresh()
                except Exception as e:
                    QMessageBox.warning(self, "Error", str(e))
            else:
                QMessageBox.warning(self, "Error", "File already exists")

    def create_folder(self, parent_dir: str):
        name, ok = QInputDialog.getText(self, "New Folder", "Folder name:")
        if ok and name:
            dir_path = os.path.join(parent_dir, name)
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path)
                    self.refresh()
                except Exception as e:
                    QMessageBox.warning(self, "Error", str(e))
            else:
                QMessageBox.warning(self, "Error", "Folder already exists")

    def rename_entry(self, file_path: str):
        name, ok = QInputDialog.getText(
            self, "Rename", "New name:", text=os.path.basename(file_path)
        )
        if ok and name:
            new_path = os.path.join(os.path.dirname(file_path), name)
            try:
                os.rename(file_path, new_path)
                self.refresh()
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    def delete_entry(self, file_path: str):
        ret = QMessageBox.question(
            self, "Delete", f"Delete {file_path}?", QMessageBox.Yes | QMessageBox.No
        )
        if ret == QMessageBox.Yes:
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                self.refresh()
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    def open_in_terminal(self, path: str):
        # Placeholder: can be connected to terminal panel later
        pass

    def open_containing_folder(self, file_path: str):
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(file_path)))


# ----------------------------------------------------------------------
# Terminal Panel
# ----------------------------------------------------------------------
class TerminalPanel(QWidget):
    command_executed = Signal(str, int)  # command, exit_code

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.project_dir = None
        self.command_history = []
        self.history_index = -1

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet(
            "QTextEdit { background-color: #1E1E1E; color: #D4D4D4; font-family: Monospace; }"
        )

        self.input = QLineEdit()
        self.input.setStyleSheet(
            "QLineEdit { background-color: #2D2D2D; color: #D4D4D4; font-family: Monospace; }"
        )
        self.input.returnPressed.connect(self.run_command)
        self.input.installEventFilter(self)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.output)
        layout.addWidget(self.input)
        self.setLayout(layout)

        self.stop_action = QAction("Stop", self)
        self.stop_action.triggered.connect(self.stop_process)
        self.addAction(self.stop_action)
        self.stop_action.setShortcut(QKeySequence("Ctrl+C"))

    def set_project_dir(self, path: str):
        self.project_dir = path
        self.output.append(
            f"<span style='color: #569CD6'>Working directory: {path}</span>"
        )

    def run_command(self, command: str = None):
        if command is None:
            command = self.input.text()
            self.input.clear()
        if not command.strip():
            return
        if not self.project_dir:
            QMessageBox.warning(self, "Error", "No project open")
            return
        self.output.append(f"<span style='color: #4EC9B0'>$ {command}</span>")
        self.process = QProcess()
        self.process.setWorkingDirectory(self.project_dir)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyRead.connect(self.read_output)
        self.process.finished.connect(self.process_finished)
        self.process.start(command)
        self.command_history.append(command)
        self.history_index = len(self.command_history)

    def read_output(self):
        if self.process:
            data = self.process.readAll()
            text = bytes(data).decode("utf-8", errors="replace")
            self.output.insertPlainText(text)
            self.output.moveCursor(QTextCursor.End)

    def process_finished(self, exit_code, exit_status):
        self.output.append(
            f"<span style='color: #C586C0'>Process exited with code {exit_code}</span>"
        )
        self.command_executed.emit(self.process.program(), exit_code)
        self.process = None

    def stop_process(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            self.process.kill()
            self.output.append("<span style='color: #F48771'>Process killed</span>")

    def eventFilter(self, obj, event):
        if obj == self.input and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Up:
                if self.history_index > 0:
                    self.history_index -= 1
                    self.input.setText(self.command_history[self.history_index])
                return True
            elif event.key() == Qt.Key_Down:
                if self.history_index < len(self.command_history) - 1:
                    self.history_index += 1
                    self.input.setText(self.command_history[self.history_index])
                else:
                    self.history_index = len(self.command_history)
                    self.input.clear()
                return True
        return super().eventFilter(obj, event)


# ----------------------------------------------------------------------
# Problems Panel
# ----------------------------------------------------------------------
class ProblemsPanel(QWidget):
    problem_activated = Signal(str, int)  # file, line

    def __init__(self, parent=None):
        super().__init__(parent)
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.list_widget)
        self.setLayout(layout)

    def set_problems(self, problems: List[Dict[str, Any]]):
        self.list_widget.clear()
        for p in problems:
            text = f"{p['file']}:{p['line']} - {p['message']}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, (p["file"], p["line"]))
            self.list_widget.addItem(item)

    def on_item_double_clicked(self, item):
        data = item.data(Qt.UserRole)
        if data:
            file_path, line = data
            self.problem_activated.emit(file_path, line)


# ----------------------------------------------------------------------
# Git Panel
# ----------------------------------------------------------------------
class GitPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.text_browser = QTextBrowser()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.text_browser)
        self.setLayout(layout)

    def set_project_dir(self, path: str):
        self.project_dir = path
        self.refresh()

    def refresh(self):
        if not hasattr(self, "project_dir") or not self.project_dir:
            self.text_browser.setPlainText("No project open")
            return
        if not os.path.isdir(os.path.join(self.project_dir, ".git")):
            self.text_browser.setPlainText("Not a Git repository")
            return
        try:
            # Git status
            status = subprocess.check_output(
                ["git", "status", "--short"],
                cwd=self.project_dir,
                stderr=subprocess.STDOUT,
                text=True,
            )
            branch = subprocess.check_output(
                ["git", "branch", "--show-current"],
                cwd=self.project_dir,
                stderr=subprocess.STDOUT,
                text=True,
            ).strip()
            text = f"<b>Branch:</b> {branch}<br><br><b>Status:</b><br>{status}"
            self.text_browser.setHtml(text)
        except subprocess.CalledProcessError as e:
            self.text_browser.setPlainText(f"Git error: {e.output}")
        except Exception as e:
            self.text_browser.setPlainText(f"Error: {e}")


# ----------------------------------------------------------------------
# Agent Tool Manager
# ----------------------------------------------------------------------
class AgentToolManager:
    """Provides tools for the agent, with security restrictions."""

    def __init__(self, project_root: str, terminal_panel: TerminalPanel = None):
        self.project_root = os.path.abspath(project_root) if project_root else None
        self.terminal_panel = terminal_panel
        self.backup_dir = (
            os.path.join(self.project_root, BACKUP_DIR_NAME)
            if self.project_root
            else None
        )
        self.modification_history = []  # list of (file_path, backup_path)
        if self.backup_dir and not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

    def set_project_root(self, path: str):
        self.project_root = os.path.abspath(path) if path else None
        self.backup_dir = (
            os.path.join(self.project_root, BACKUP_DIR_NAME)
            if self.project_root
            else None
        )
        self.modification_history.clear()
        if self.backup_dir and not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

    def _safe_path(self, file_path: str) -> str:
        """Ensure path is within project root."""
        if not self.project_root:
            raise PermissionError("No project root set")
        abs_path = os.path.abspath(file_path)
        if not abs_path.startswith(self.project_root):
            raise PermissionError(f"Access denied to {abs_path}")
        return abs_path

    def _backup_file(self, file_path: str):
        if not self.backup_dir:
            return
        rel_path = os.path.relpath(file_path, self.project_root)
        backup_path = os.path.join(self.backup_dir, rel_path + ".bak")
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(file_path, backup_path)
        self.modification_history.append((file_path, backup_path))

    def list_files(self, path: str = ".") -> List[str]:
        """List files in directory relative to project root."""
        if not self.project_root:
            return []
        target = self._safe_path(os.path.join(self.project_root, path))
        files = []
        for root, dirs, filenames in os.walk(target):
            # Skip hidden dirs
            dirs[:] = [d for d in dirs if d not in HIDDEN_DIRS]
            for f in filenames:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, self.project_root)
                files.append(rel)
        return files

    def read_file(self, file_path: str) -> str:
        path = self._safe_path(os.path.join(self.project_root, file_path))
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def write_file(self, file_path: str, content: str) -> str:
        path = self._safe_path(os.path.join(self.project_root, file_path))
        self._backup_file(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File written: {file_path}"

    def patch_file(self, file_path: str, old_text: str, new_text: str) -> str:
        path = self._safe_path(os.path.join(self.project_root, file_path))
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if old_text not in content:
            return f"Error: old_text not found in {file_path}"
        self._backup_file(path)
        new_content = content.replace(old_text, new_text, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Patched {file_path}"

    def search_project(self, query: str) -> List[Dict[str, Any]]:
        if not self.project_root:
            return []
        results = []
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in HIDDEN_DIRS]
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) > 1_000_000:  # skip large files
                    continue
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if query.lower() in line.lower():
                                rel_path = os.path.relpath(file_path, self.project_root)
                                results.append(
                                    {
                                        "file": rel_path,
                                        "line": i,
                                        "content": line.strip(),
                                    }
                                )
                except:
                    pass
        return results

    def run_command(self, command: str, timeout: int = 30) -> str:
        if not self.project_root:
            return "Error: No project root"
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return f"Exit code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out"
        except Exception as e:
            return f"Error: {e}"

    def run_python(self, file_path: str, args: str = "") -> str:
        if not self.project_root:
            return "Error: No project root"
        path = self._safe_path(os.path.join(self.project_root, file_path))
        cmd = f"python {path} {args}"
        return self.run_command(cmd)

    def git_status(self) -> str:
        return self.run_command("git status --short")

    def git_diff(self) -> str:
        return self.run_command("git diff")

    def get_recent_modifications(self) -> List[Tuple[str, str]]:
        return self.modification_history.copy()

    def revert_modification(self, index: int) -> str:
        if 0 <= index < len(self.modification_history):
            file_path, backup_path = self.modification_history[index]
            shutil.copy2(backup_path, file_path)
            # Remove from history
            self.modification_history.pop(index)
            return f"Reverted {file_path}"
        return "Invalid modification index"

    def get_diff(self, file_path: str) -> str:
        path = self._safe_path(os.path.join(self.project_root, file_path))
        # Find latest backup for this file
        backups = [b for f, b in self.modification_history if f == path]
        if not backups:
            return "No backup found"
        backup_path = backups[-1]
        with open(backup_path, "r", encoding="utf-8", errors="replace") as f:
            old = f.read()
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            new = f.read()
        diff = difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="")
        return "\n".join(diff)


# ----------------------------------------------------------------------
# Agent Client
# ----------------------------------------------------------------------
class AgentClient(QThread):
    response_ready = Signal(str)
    error_occurred = Signal(str)
    tool_executed = Signal(str, str)  # tool name, result

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        tools: List[Dict],
        system_prompt: str,
    ):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.tools = tools
        self.system_prompt = system_prompt
        self.messages = []
        self.is_running = True

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def run(self):
        try:
            while self.is_running:
                response = self._call_api()
                if not response:
                    return
                if response.get("choices"):
                    choice = response["choices"][0]
                    message = choice["message"]
                    if message.get("content"):
                        self.response_ready.emit(message["content"])
                    if message.get("tool_calls"):
                        for tool_call in message["tool_calls"]:
                            tool_name = tool_call["function"]["name"]
                            arguments = json.loads(tool_call["function"]["arguments"])
                            result = self._execute_tool(tool_name, arguments)
                            self.tool_executed.emit(tool_name, result)
                            self.messages.append(
                                {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [tool_call],
                                }
                            )
                            self.messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "content": result,
                                }
                            )
                        continue  # continue loop to get next response
                    else:
                        return
                else:
                    return
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _call_api(self):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt}]
            + self.messages,
            "temperature": self.temperature,
            "tools": self.tools,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        # Delegate to AgentToolManager
        # This method will be overridden or connected externally.
        # Placeholder: we'll set a callback function.
        if hasattr(self, "tool_executor"):
            return self.tool_executor(tool_name, arguments)
        return f"Tool '{tool_name}' not implemented"

    def stop(self):
        self.is_running = False


class BrainAgentClient(QThread):
    """Run the canonical agent brain without duplicating provider logic."""

    response_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, objective: str):
        super().__init__()
        self.objective = objective
        self.is_running = True

    def run(self):
        try:
            if self.is_running:
                self.response_ready.emit(ask_model(self.objective, verbose=True))
        except Exception as error:
            self.error_occurred.emit(str(error))

    def stop(self):
        self.is_running = False


# ----------------------------------------------------------------------
# Agent Panel
# ----------------------------------------------------------------------
class AgentPanel(QWidget):
    def __init__(self, settings: QSettings, project_manager, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.project_manager = project_manager
        self.agent_thread = None
        self.tool_manager = None

        self.chat_history = QTextBrowser()
        self.chat_history.setOpenExternalLinks(True)
        self.chat_history.setStyleSheet(
            "QTextBrowser { background-color: #1E1E1E; color: #D4D4D4; font-family: Monospace; }"
        )

        self.input = QTextEdit()
        self.input.setFixedHeight(60)
        self.input.setStyleSheet(
            "QTextEdit { background-color: #2D2D2D; color: #D4D4D4; font-family: Monospace; }"
        )

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input)
        input_layout.addWidget(self.send_button)

        layout = QVBoxLayout()
        layout.addWidget(self.chat_history)
        layout.addLayout(input_layout)
        self.setLayout(layout)

        self.status_label = QLabel("Agent: idle")
        layout.addWidget(self.status_label)

    def set_project(self, project_path: str):
        self.tool_manager = AgentToolManager(project_path)
        self.project_manager.tool_manager = self.tool_manager  # share for revert

    def send_message(self):
        if not self.project_manager.current_project:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return
        if not self.tool_manager:
            self.set_project(self.project_manager.current_project)

        user_text = self.input.toPlainText().strip()
        if not user_text:
            return
        self.input.clear()
        self.chat_history.append(f"<b style='color: #569CD6'>You:</b> {user_text}")

        self.agent_thread = BrainAgentClient(user_text)
        self.agent_thread.response_ready.connect(self.on_response)
        self.agent_thread.error_occurred.connect(self.on_error)
        self.agent_thread.finished.connect(self.on_finished)
        self.agent_thread.start()
        self.status_label.setText("Agent: working...")
        return

        system_prompt = """You are a coding assistant working inside a local IDE. You have access to tools to inspect and modify the project.
        Always explain your reasoning before making changes. Make minimal changes. Never modify files outside the project root.
        Use the tools provided to accomplish the user's request. When you are done, summarize what you changed."""

        # Define tools schema
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory (relative to project root)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path relative to project root, default '.'",
                            }
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from the project",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file (creates or overwrites)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["file_path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "patch_file",
                    "description": "Replace a specific string in a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "old_text": {"type": "string"},
                            "new_text": {"type": "string"},
                        },
                        "required": ["file_path", "old_text", "new_text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_project",
                    "description": "Search for a query in project files",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Execute a shell command in the project directory",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_python",
                    "description": "Run a Python file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "args": {
                                "type": "string",
                                "description": "Optional command line arguments",
                            },
                        },
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_status",
                    "description": "Show git status",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_diff",
                    "description": "Show git diff",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        self.agent_thread = AgentClient(
            base_url, api_key, model, temperature, tools, system_prompt
        )
        self.agent_thread.tool_executor = self.execute_tool
        self.agent_thread.add_user_message(user_text)
        self.agent_thread.response_ready.connect(self.on_response)
        self.agent_thread.error_occurred.connect(self.on_error)
        self.agent_thread.finished.connect(self.on_finished)
        self.agent_thread.start()
        self.status_label.setText("Agent: working...")

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        if not self.tool_manager:
            return "No project"
        # Ask for confirmation on destructive commands
        if tool_name == "run_command":
            cmd = args.get("command", "")
            if self.is_dangerous(cmd):
                ret = QMessageBox.question(
                    self,
                    "Confirm Command",
                    f"Agent wants to run: {cmd}\nAllow?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if ret != QMessageBox.Yes:
                    return "User denied command execution"
        if tool_name == "write_file":
            ret = QMessageBox.question(
                self,
                "Confirm File Write",
                f"Agent wants to write to {args.get('file_path')}\nAllow?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return "User denied file write"

        tool_map = {
            "list_files": lambda: self.tool_manager.list_files(args.get("path", ".")),
            "read_file": lambda: self.tool_manager.read_file(args["file_path"]),
            "write_file": lambda: self.tool_manager.write_file(
                args["file_path"], args["content"]
            ),
            "patch_file": lambda: self.tool_manager.patch_file(
                args["file_path"], args["old_text"], args["new_text"]
            ),
            "search_project": lambda: self.tool_manager.search_project(args["query"]),
            "run_command": lambda: self.tool_manager.run_command(args["command"]),
            "run_python": lambda: self.tool_manager.run_python(
                args["file_path"], args.get("args", "")
            ),
            "git_status": lambda: self.tool_manager.git_status(),
            "git_diff": lambda: self.tool_manager.git_diff(),
        }
        func = tool_map.get(tool_name)
        if func:
            try:
                result = func()
                if isinstance(result, list):
                    result = json.dumps(result, indent=2)
                return str(result)
            except Exception as e:
                return f"Error executing tool: {e}"
        return f"Unknown tool: {tool_name}"

    def is_dangerous(self, command: str) -> bool:
        dangerous_patterns = [
            r"\brm\b",
            r"\bmv\b",
            r"\bcp\b",
            r"\bdd\b",
            r"\bmkfs\b",
            r"\bformat\b",
            r"\bdel\b",
            r"\berase\b",
            r"\bkill\b",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return True
        return False

    def on_response(self, text: str):
        self.chat_history.append(f"<b style='color: #4EC9B0'>Agent:</b> {text}")
        self.chat_history.moveCursor(QTextCursor.End)

    def on_error(self, error: str):
        self.chat_history.append(f"<span style='color: #F48771'>Error: {error}</span>")
        self.status_label.setText("Agent: error")

    def on_finished(self):
        self.status_label.setText("Agent: idle")
        self.agent_thread = None


# ----------------------------------------------------------------------
# Project Manager
# ----------------------------------------------------------------------
class ProjectManager(QObject):
    project_changed = Signal(str)

    def __init__(self, settings: QSettings):
        super().__init__()
        self.settings = settings
        self.current_project = None
        self.tool_manager = None
        self.recent_projects = self.settings.value("recent_projects", [], type=list)

    def open_project(self, path: str):
        if path and os.path.isdir(path):
            self.current_project = path
            # Update recent list
            if path in self.recent_projects:
                self.recent_projects.remove(path)
            self.recent_projects.insert(0, path)
            self.recent_projects = self.recent_projects[:10]
            self.settings.setValue("recent_projects", self.recent_projects)
            self.project_changed.emit(path)

    def close_project(self):
        self.current_project = None
        self.project_changed.emit("")


# ----------------------------------------------------------------------
# Settings Dialog
# ----------------------------------------------------------------------
class SettingsDialog(QDialog):
    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Settings")
        self.setModal(True)

        layout = QVBoxLayout()
        form = QFormLayout()

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(float(settings.value("agent/temperature", 0.2)))

        self.python_exec = QLineEdit(settings.value("env/python_exec", "python"))
        self.terminal_shell = QLineEdit(
            settings.value("env/terminal_shell", os.environ.get("SHELL", "/bin/bash"))
        )
        self.font_family = QLineEdit(settings.value("editor/font_family", "Monospace"))
        self.font_size = QSpinBox()
        self.font_size.setRange(8, 72)
        self.font_size.setValue(int(settings.value("editor/font_size", 10)))

        self.hidden_folders = QLineEdit(
            settings.value("explorer/hidden_folders", ",".join(HIDDEN_DIRS))
        )

        form.addRow("Temperature:", self.temperature)
        form.addRow("Python Executable:", self.python_exec)
        form.addRow("Terminal Shell:", self.terminal_shell)
        form.addRow("Editor Font:", self.font_family)
        form.addRow("Font Size:", self.font_size)
        form.addRow("Hidden Folders (comma-sep):", self.hidden_folders)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def save_settings(self):
        self.settings.setValue("agent/temperature", self.temperature.value())
        self.settings.setValue("env/python_exec", self.python_exec.text().strip())
        self.settings.setValue("env/terminal_shell", self.terminal_shell.text().strip())
        self.settings.setValue("editor/font_family", self.font_family.text().strip())
        self.settings.setValue("editor/font_size", self.font_size.value())
        self.settings.setValue(
            "explorer/hidden_folders", self.hidden_folders.text().strip()
        )


# ----------------------------------------------------------------------
# Main Window
# ----------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.project_manager = ProjectManager(self.settings)
        self.setWindowTitle(APP_NAME)
        self.resize(1200, 800)

        # Set dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1E1E1E; color: #D4D4D4; }
            QMenuBar, QMenu { background-color: #2D2D2D; color: #D4D4D4; }
            QMenuBar::item:selected, QMenu::item:selected { background-color: #3E3E3E; }
            QTabWidget::pane { border: 1px solid #3E3E3E; }
            QTabBar::tab { background: #2D2D2D; color: #D4D4D4; padding: 6px; }
            QTabBar::tab:selected { background: #1E1E1E; }
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #2D2D2D; color: #D4D4D4; border: 1px solid #3E3E3E;
            }
            QPushButton { background-color: #0E639C; color: white; border: none; padding: 5px; }
            QPushButton:hover { background-color: #1177BB; }
            QLabel { color: #D4D4D4; }
            QTreeView { background-color: #252526; color: #D4D4D4; }
            QListWidget { background-color: #252526; color: #D4D4D4; }
            QStatusBar { background-color: #007ACC; color: white; }
        """)

        # Create central widget and layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        # Left panel: Project Explorer
        self.project_explorer = ProjectExplorer()
        self.project_explorer.open_file_requested.connect(self.open_file)

        # Center: Editor Tabs
        self.editor_tabs = EditorTabs()
        self.editor_tabs.new_editor()  # start with blank

        # Bottom panel: Terminal, Problems, Git
        self.bottom_tabs = QTabWidget()
        self.terminal_panel = TerminalPanel()
        self.problems_panel = ProblemsPanel()
        self.git_panel = GitPanel()
        self.bottom_tabs.addTab(self.terminal_panel, "Terminal")
        self.bottom_tabs.addTab(self.problems_panel, "Problems")
        self.bottom_tabs.addTab(self.git_panel, "Git")

        # Right panel: Agent
        self.agent_panel = AgentPanel(self.settings, self.project_manager)

        # Splitters
        left_splitter = QSplitter(Qt.Horizontal)
        left_splitter.addWidget(self.project_explorer)
        left_splitter.addWidget(self.editor_tabs)
        left_splitter.addWidget(self.agent_panel)
        left_splitter.setChildrenCollapsible(False)
        left_splitter.setHandleWidth(6)
        self.project_explorer.setMinimumWidth(230)
        self.editor_tabs.setMinimumWidth(480)
        self.agent_panel.setMinimumWidth(300)
        left_splitter.setStretchFactor(0, 0)
        left_splitter.setStretchFactor(1, 1)
        left_splitter.setStretchFactor(2, 0)
        left_splitter.setSizes([250, 600, 350])

        vertical_splitter = QSplitter(Qt.Vertical)
        vertical_splitter.addWidget(left_splitter)
        vertical_splitter.addWidget(self.bottom_tabs)
        vertical_splitter.setChildrenCollapsible(False)
        vertical_splitter.setHandleWidth(6)
        vertical_splitter.setStretchFactor(0, 1)
        vertical_splitter.setStretchFactor(1, 0)
        vertical_splitter.setSizes([600, 200])

        main_layout.addWidget(vertical_splitter)

        # Menu
        self.create_menus()

        # Status bar
        self.statusBar().showMessage("Ready")

        # Connect signals
        self.project_manager.project_changed.connect(self.on_project_changed)
        self.editor_tabs.currentChanged.connect(self.on_editor_changed)
        self.terminal_panel.command_executed.connect(self.on_command_executed)
        self.problems_panel.problem_activated.connect(self.open_file_at_line)

        # Load recent projects if any
        self.update_recent_projects_menu()
        if self.project_manager.recent_projects:
            last_project = self.project_manager.recent_projects[0]
            self.project_manager.open_project(last_project)

    def create_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        open_action = QAction("Open Project...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.open_project_dialog)
        file_menu.addAction(open_action)

        recent_menu = file_menu.addMenu("Recent Projects")
        self.recent_menu = recent_menu

        file_menu.addSeparator()
        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.save_current_file)
        file_menu.addAction(save_action)

        save_all_action = QAction("Save All", self)
        save_all_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_all_action.triggered.connect(self.editor_tabs.save_all)
        file_menu.addAction(save_all_action)

        file_menu.addSeparator()
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menubar.addMenu("&Edit")
        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence("Ctrl+Y"))
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()
        find_action = QAction("Find", self)
        find_action.setShortcut(QKeySequence("Ctrl+F"))
        find_action.triggered.connect(self.show_find)
        edit_menu.addAction(find_action)

        replace_action = QAction("Replace", self)
        replace_action.setShortcut(QKeySequence("Ctrl+H"))
        replace_action.triggered.connect(self.show_replace)
        edit_menu.addAction(replace_action)

        run_menu = menubar.addMenu("&Run")
        run_file_action = QAction("Run Python File", self)
        run_file_action.setShortcut(QKeySequence("F5"))
        run_file_action.triggered.connect(self.run_current_python)
        run_menu.addAction(run_file_action)

        run_tests_action = QAction("Run Tests", self)
        run_tests_action.triggered.connect(self.run_tests)
        run_menu.addAction(run_tests_action)

        debug_menu = menubar.addMenu("&Debug")
        debug_file_action = QAction("Debug Python File (pdb)", self)
        debug_file_action.triggered.connect(self.debug_current_python)
        debug_menu.addAction(debug_file_action)

        view_menu = menubar.addMenu("&View")
        refresh_action = QAction("Refresh Project", self)
        refresh_action.triggered.connect(self.project_explorer.refresh)
        view_menu.addAction(refresh_action)

    def open_project_dialog(self):
        path = QFileDialog.getExistingDirectory(
            self, "Open Project", os.path.expanduser("~")
        )
        if path:
            self.project_manager.open_project(path)

    def update_recent_projects_menu(self):
        self.recent_menu.clear()
        if not self.project_manager.recent_projects:
            no_recent = QAction("No recent projects", self)
            no_recent.setEnabled(False)
            self.recent_menu.addAction(no_recent)
        else:
            for path in self.project_manager.recent_projects:
                action = QAction(path, self)
                action.triggered.connect(
                    lambda checked, p=path: self.project_manager.open_project(p)
                )
                self.recent_menu.addAction(action)

    def on_project_changed(self, path: str):
        if path:
            selected_root = Path(path).expanduser().resolve()
            agent_tools.WORKSPACE = selected_root
            agent_brain.PROJECT_WORKSPACE = str(selected_root)
            self.project_explorer.set_project(path)
            self.terminal_panel.set_project_dir(path)
            self.git_panel.set_project_dir(path)
            self.agent_panel.set_project(path)
            self.statusBar().showMessage(f"Project: {path}")
            self.update_recent_projects_menu()
        else:
            self.project_explorer.set_project("")
            self.terminal_panel.set_project_dir("")
            self.git_panel.set_project_dir("")
            self.statusBar().showMessage("No project open")

    def open_file(self, file_path: str):
        self.editor_tabs.open_file(file_path)

    def open_file_at_line(self, file_path: str, line: int):
        editor = self.editor_tabs.open_file(file_path)
        if editor:
            cursor = editor.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.movePosition(QTextCursor.Down, QTextCursor.MoveAnchor, line - 1)
            editor.setTextCursor(cursor)
            editor.setFocus()

    def save_current_file(self):
        self.editor_tabs.save_current()

    def on_editor_changed(self, index):
        editor = self.editor_tabs.current_editor()
        if editor:
            self.setWindowModified(editor.is_modified)

    def undo(self):
        editor = self.editor_tabs.current_editor()
        if editor:
            editor.undo()

    def redo(self):
        editor = self.editor_tabs.current_editor()
        if editor:
            editor.redo()

    def show_find(self):
        editor = self.editor_tabs.current_editor()
        if editor:
            # Implement simple search dialog
            search_text, ok = QInputDialog.getText(self, "Find", "Search for:")
            if ok and search_text:
                cursor = editor.textCursor()
                if not editor.find(search_text):
                    QMessageBox.information(self, "Find", "Text not found")

    def show_replace(self):
        editor = self.editor_tabs.current_editor()
        if editor:
            search_text, ok = QInputDialog.getText(self, "Replace", "Find:")
            if ok and search_text:
                replace_text, ok2 = QInputDialog.getText(
                    self, "Replace", "Replace with:"
                )
                if ok2:
                    content = editor.toPlainText()
                    new_content = content.replace(search_text, replace_text)
                    if new_content != content:
                        editor.setPlainText(new_content)
                    else:
                        QMessageBox.information(self, "Replace", "Text not found")

    def run_current_python(self):
        editor = self.editor_tabs.current_editor()
        if editor and editor.file_path and editor.file_path.endswith(".py"):
            if not self.project_manager.current_project:
                QMessageBox.warning(self, "No Project", "Open a project first.")
                return
            self.terminal_panel.input.setText(f'python "{editor.file_path}"')
            self.terminal_panel.run_command()

    def run_tests(self):
        if not self.project_manager.current_project:
            return
        self.terminal_panel.input.setText("python -m unittest discover")
        self.terminal_panel.run_command()

    def debug_current_python(self):
        editor = self.editor_tabs.current_editor()
        if editor and editor.file_path and editor.file_path.endswith(".py"):
            if not self.project_manager.current_project:
                return
            self.terminal_panel.input.setText(f'python -m pdb "{editor.file_path}"')
            self.terminal_panel.run_command()

    def on_command_executed(self, command: str, exit_code: int):
        # Parse output for problems if command was python-related
        if "python" in command:
            self.parse_python_problems()

    def parse_python_problems(self):
        # Basic parser for Python tracebacks from terminal output
        output = self.terminal_panel.output.toPlainText()
        problems = []
        # Regex for file paths and line numbers in traceback
        pattern = r'File "([^"]+)", line (\d+)'
        matches = re.findall(pattern, output)
        for m in matches[-3:]:  # last few traceback entries
            problems.append(
                {"file": m[0], "line": int(m[1]), "message": "Traceback location"}
            )
        if problems:
            self.problems_panel.set_problems(problems)
            self.bottom_tabs.setCurrentWidget(self.problems_panel)

    def show_settings(self):
        global HIDDEN_DIRS
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            dialog.save_settings()
            # Apply font settings
            font = QFont(self.settings.value("editor/font_family", "Monospace"))
            font.setPointSize(int(self.settings.value("editor/font_size", 10)))
            for editor in self.editor_tabs.editors:
                editor.setFont(font)
            # Update hidden folders
            hidden = set(
                self.settings.value(
                    "explorer/hidden_folders", ",".join(HIDDEN_DIRS)
                ).split(",")
            )
            HIDDEN_DIRS = hidden

    def closeEvent(self, event: QCloseEvent):
        # Check unsaved files
        if self.editor_tabs.editors:
            unsaved = [e for e in self.editor_tabs.editors if e.is_modified]
            if unsaved:
                ret = QMessageBox.question(
                    self,
                    "Unsaved Changes",
                    f"You have {len(unsaved)} unsaved file(s). Save all?",
                    QMessageBox.SaveAll | QMessageBox.Discard | QMessageBox.Cancel,
                )
                if ret == QMessageBox.SaveAll:
                    self.editor_tabs.save_all()
                    event.accept()
                elif ret == QMessageBox.Discard:
                    event.accept()
                else:
                    event.ignore()
                    return
        # Stop agent thread if running
        if self.agent_panel.agent_thread and self.agent_panel.agent_thread.isRunning():
            self.agent_panel.agent_thread.stop()
            self.agent_panel.agent_thread.wait(1000)
        event.accept()


# ----------------------------------------------------------------------
# Main Entry
# ----------------------------------------------------------------------
def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
