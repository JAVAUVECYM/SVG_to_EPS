# 文件名: svg_to_eps_comfyui.py
# 保存位置: ComfyUI/custom_nodes/svg_to_eps_node/

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, List
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps

# ComfyUI imports
import folder_paths
from nodes import MAX_RESOLUTION, SaveImage
import torch

# SVG处理库
try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False
    print("⚠️ CairoSVG未安装，如需使用请运行: pip install cairosvg")

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPS
    SVGLIB_AVAILABLE = True
except ImportError:
    SVGLIB_AVAILABLE = False
    print("⚠️ svglib未安装，如需使用请运行: pip install svglib reportlab")


class SVGEPSConverterCore:
    """SVG到EPS转换核心类"""
    
    @staticmethod
    def check_inkscape() -> bool:
        """检查Inkscape是否可用"""
        try:
            result = subprocess.run(['inkscape', '--version'], 
                                  capture_output=True, text=True, shell=False)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            return False
    
    @classmethod
    def get_available_methods(cls) -> Dict[str, bool]:
        """获取可用的转换方法"""
        return {
            "inkscape": cls.check_inkscape(),
            "cairosvg": CAIROSVG_AVAILABLE,
            "svglib": SVGLIB_AVAILABLE
        }
    
    @classmethod
    def convert_with_inkscape(cls, svg_path: Path, eps_path: Path, 
                            text_to_path: bool = True, 
                            dpi: int = 300) -> Tuple[bool, str]:
        """使用Inkscape转换（最佳质量）"""
        try:
            cmd = [
                'inkscape',
                '--export-type=eps',
                f'--export-filename={eps_path}',
                f'--export-dpi={dpi}',
                '--export-area-drawing',
                '--export-overwrite',
            ]
            
            if text_to_path:
                cmd.append('--export-text-to-path')
            
            cmd.append(str(svg_path))
            
            result = subprocess.run(cmd, capture_output=True, text=True, shell=False, timeout=30)
            
            if result.returncode == 0:
                return True, f"Inkscape转换成功: {eps_path.name}"
            else:
                error_msg = result.stderr[:200] if result.stderr else "未知错误"
                return False, f"Inkscape错误: {error_msg}"
                
        except subprocess.TimeoutExpired:
            return False, "Inkscape转换超时"
        except Exception as e:
            return False, f"执行错误: {str(e)}"
    
    @classmethod
    def convert_with_cairosvg(cls, svg_path: Path, eps_path: Path, 
                            dpi: int = 300) -> Tuple[bool, str]:
        """使用CairoSVG转换"""
        if not CAIROSVG_AVAILABLE:
            return False, "CairoSVG未安装，请运行: pip install cairosvg"
        
        try:
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            
            cairosvg.svg2eps(
                bytestring=svg_content.encode('utf-8'),
                write_to=str(eps_path),
                dpi=dpi
            )
            return True, f"CairoSVG转换成功: {eps_path.name}"
            
        except Exception as e:
            return False, f"CairoSVG异常: {str(e)}"
    
    @classmethod
    def convert_with_svglib(cls, svg_path: Path, eps_path: Path) -> Tuple[bool, str]:
        """使用svglib转换"""
        if not SVGLIB_AVAILABLE:
            return False, "svglib未安装，请运行: pip install svglib reportlab"
        
        try:
            drawing = svg2rlg(str(svg_path))
            
            if drawing is None:
                return False, "无法解析SVG文件"
            
            with open(eps_path, 'wb') as f:
                renderPS.drawToFile(drawing, f, 'EPS')
            
            return True, f"svglib转换成功: {eps_path.name}"
            
        except Exception as e:
            return False, f"svglib异常: {str(e)}"


class SVGToEPSNode:
    """SVG到EPS转换节点"""
    
    def __init__(self):
        self.output_dir = Path(folder_paths.get_output_directory()) / "svg_to_eps"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_file": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "输入SVG文件名（在input目录中）"
                }),
            },
            "optional": {
                "method": (["auto", "inkscape", "cairosvg", "svglib"], {
                    "default": "auto"
                }),
                "dpi": ("INT", {
                    "default": 300,
                    "min": 72,
                    "max": 1200,
                    "step": 1
                }),
                "text_to_path": (["enable", "disable"], {
                    "default": "enable"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("eps_path", "status")
    FUNCTION = "convert_svg_to_eps"
    CATEGORY = "image/conversion"
    OUTPUT_NODE = True
    
    def convert_svg_to_eps(self, svg_file: str, method: str = "auto", 
                          dpi: int = 300, text_to_path: str = "enable") -> Tuple[str, str]:
        """执行SVG到EPS转换"""
        
        if not svg_file.strip():
            return ("", "❌ 错误: SVG文件名不能为空")
        
        # 查找SVG文件
        svg_path = self._find_svg_file(svg_file)
        if svg_path is None:
            return ("", f"❌ 错误: 找不到SVG文件 '{svg_file}'")
        
        # 验证文件格式
        if svg_path.suffix.lower() != '.svg':
            return ("", f"❌ 错误: 文件不是SVG格式: {svg_path.suffix}")
        
        # 检查依赖
        deps = SVGEPSConverterCore.get_available_methods()
        
        # 自动选择最佳方法
        if method == "auto":
            if deps["inkscape"]:
                method = "inkscape"
            elif deps["cairosvg"]:
                method = "cairosvg"
            elif deps["svglib"]:
                method = "svglib"
            else:
                return ("", "❌ 错误: 没有可用的转换工具，请安装Inkscape或相关Python库")
        
        # 检查所选方法是否可用
        if method == "inkscape" and not deps["inkscape"]:
            return ("", "❌ 错误: Inkscape不可用，请安装Inkscape或选择其他方法")
        elif method == "cairosvg" and not deps["cairosvg"]:
            return ("", "❌ 错误: CairoSVG不可用，请运行: pip install cairosvg")
        elif method == "svglib" and not deps["svglib"]:
            return ("", "❌ 错误: svglib不可用，请运行: pip install svglib reportlab")
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eps_filename = f"{svg_path.stem}_{timestamp}.eps"
        eps_path = self.output_dir / eps_filename
        
        # 避免文件名冲突
        counter = 1
        while eps_path.exists():
            eps_filename = f"{svg_path.stem}_{timestamp}_{counter:03d}.eps"
            eps_path = self.output_dir / eps_filename
            counter += 1
        
        # 执行转换
        text_to_path_bool = (text_to_path == "enable")
        
        if method == "inkscape":
            success, message = SVGEPSConverterCore.convert_with_inkscape(
                svg_path, eps_path, text_to_path_bool, dpi
            )
        elif method == "cairosvg":
            success, message = SVGEPSConverterCore.convert_with_cairosvg(svg_path, eps_path, dpi)
        elif method == "svglib":
            success, message = SVGEPSConverterCore.convert_with_svglib(svg_path, eps_path)
        else:
            return ("", f"❌ 错误: 未知的转换方法: {method}")
        
        # 返回结果
        if success:
            if eps_path.exists() and eps_path.stat().st_size > 0:
                return (str(eps_path), f"✅ {message}")
            else:
                return ("", "⚠️ 转换成功但输出文件为空")
        else:
            return ("", f"❌ {message}")
    
    def _find_svg_file(self, filename: str) -> Optional[Path]:
        """查找SVG文件"""
        # 检查是否为绝对路径
        svg_path = Path(filename)
        if svg_path.is_absolute() and svg_path.exists():
            return svg_path
        
        # 在输入目录中查找
        input_dir = Path(folder_paths.get_input_directory())
        possible_paths = [
            input_dir / filename,
            input_dir / f"{filename}.svg",
            input_dir / f"{filename}.SVG",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # 在输出目录中查找
        output_dir = Path(folder_paths.get_output_directory())
        possible_paths = [
            output_dir / filename,
            output_dir / f"{filename}.svg",
            output_dir / f"{filename}.SVG",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None


class BatchSVGtoEPSNode:
    """批量SVG到EPS转换节点"""
    
    def __init__(self):
        self.output_dir = Path(folder_paths.get_output_directory()) / "batch_svg_to_eps"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_directory": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "输入SVG文件目录路径"
                }),
            },
            "optional": {
                "method": (["auto", "inkscape", "cairosvg", "svglib"], {
                    "default": "auto"
                }),
                "dpi": ("INT", {
                    "default": 300,
                    "min": 72,
                    "max": 1200,
                    "step": 1
                }),
                "text_to_path": (["enable", "disable"], {
                    "default": "enable"
                }),
                "recursive": ("BOOLEAN", {
                    "default": False
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_directory", "summary")
    FUNCTION = "batch_convert"
    CATEGORY = "image/conversion"
    OUTPUT_NODE = True
    
    def batch_convert(self, input_directory: str, method: str = "auto",
                     dpi: int = 300, text_to_path: str = "enable", 
                     recursive: bool = False) -> Tuple[str, str]:
        """批量转换SVG文件"""
        
        if not input_directory.strip():
            return ("", "❌ 错误: 输入目录不能为空")
        
        # 查找输入目录
        input_dir = self._find_directory(input_directory)
        if input_dir is None:
            return ("", f"❌ 错误: 找不到目录 '{input_directory}'")
        
        # 查找SVG文件
        if recursive:
            svg_files = list(input_dir.rglob("*.svg")) + list(input_dir.rglob("*.SVG"))
        else:
            svg_files = list(input_dir.glob("*.svg")) + list(input_dir.glob("*.SVG"))
        
        if not svg_files:
            return ("", f"❌ 错误: 在 '{input_directory}' 中找不到SVG文件")
        
        # 创建输出子目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = self.output_dir / f"batch_{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # 转换统计
        stats = {
            "total": len(svg_files),
            "success": 0,
            "failed": 0,
            "failed_list": []
        }
        
        # 批量转换
        for svg_file in svg_files:
            # 保持相对路径结构
            if svg_file.parent != input_dir:
                rel_path = svg_file.relative_to(input_dir).parent
                output_subdir = batch_dir / rel_path
                output_subdir.mkdir(parents=True, exist_ok=True)
            else:
                output_subdir = batch_dir
            
            # 生成输出文件名
            eps_filename = f"{svg_file.stem}.eps"
            eps_path = output_subdir / eps_filename
            
            # 避免文件名冲突
            counter = 1
            while eps_path.exists():
                eps_filename = f"{svg_file.stem}_{counter:03d}.eps"
                eps_path = output_subdir / eps_filename
                counter += 1
            
            # 执行转换
            text_to_path_bool = (text_to_path == "enable")
            
            success, message = self._convert_single_file(
                svg_file, eps_path, method, text_to_path_bool, dpi
            )
            
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
                stats["failed_list"].append({
                    "file": svg_file.name,
                    "error": message[:100]
                })
        
        # 生成摘要
        summary = self._generate_summary(stats, batch_dir)
        
        return (str(batch_dir), summary)
    
    def _find_directory(self, directory: str) -> Optional[Path]:
        """查找目录"""
        # 检查是否为绝对路径
        dir_path = Path(directory)
        if dir_path.is_absolute() and dir_path.is_dir():
            return dir_path
        
        # 在输入目录中查找
        input_dir = Path(folder_paths.get_input_directory())
        possible_path = input_dir / directory
        if possible_path.exists() and possible_path.is_dir():
            return possible_path
        
        # 在输出目录中查找
        output_dir = Path(folder_paths.get_output_directory())
        possible_path = output_dir / directory
        if possible_path.exists() and possible_path.is_dir():
            return possible_path
        
        return None
    
    def _convert_single_file(self, svg_path: Path, eps_path: Path, 
                           method: str, text_to_path: bool, dpi: int) -> Tuple[bool, str]:
        """转换单个文件"""
        deps = SVGEPSConverterCore.get_available_methods()
        
        # 自动选择方法
        if method == "auto":
            if deps["inkscape"]:
                method = "inkscape"
            elif deps["cairosvg"]:
                method = "cairosvg"
            elif deps["svglib"]:
                method = "svglib"
            else:
                return False, "无可用转换工具"
        
        # 执行转换
        if method == "inkscape" and deps["inkscape"]:
            return SVGEPSConverterCore.convert_with_inkscape(svg_path, eps_path, text_to_path, dpi)
        elif method == "cairosvg" and deps["cairosvg"]:
            return SVGEPSConverterCore.convert_with_cairosvg(svg_path, eps_path, dpi)
        elif method == "svglib" and deps["svglib"]:
            return SVGEPSConverterCore.convert_with_svglib(svg_path, eps_path)
        else:
            return False, f"方法不可用: {method}"
    
    def _generate_summary(self, stats: Dict, output_dir: Path) -> str:
        """生成转换摘要"""
        lines = [
            "📊 批量转换完成",
            "=" * 40,
            f"📁 输出目录: {output_dir.name}",
            f"📄 总文件数: {stats['total']}",
            f"✅ 成功: {stats['success']}",
            f"❌ 失败: {stats['failed']}",
        ]
        
        if stats['total'] > 0:
            success_rate = stats['success'] / stats['total'] * 100
            lines.append(f"📈 成功率: {success_rate:.1f}%")
        
        if stats['failed'] > 0:
            lines.extend([
                "",
                "📝 失败文件列表:",
                "-" * 30
            ])
            for i, fail in enumerate(stats['failed_list'][:5], 1):
                lines.append(f"{i}. {fail['file']}: {fail['error']}")
            
            if len(stats['failed_list']) > 5:
                lines.append(f"... 还有 {len(stats['failed_list']) - 5} 个失败文件")
        
        return "\n".join(lines)


class EPSToImageNode:
    """EPS到图像转换节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "eps_file": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "输入EPS文件路径"
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": MAX_RESOLUTION,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": MAX_RESOLUTION,
                    "step": 8
                }),
            },
            "optional": {
                "background_color": (["white", "black", "transparent"], {
                    "default": "white"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "convert_eps_to_image"
    CATEGORY = "image/conversion"
    
    def convert_eps_to_image(self, eps_file: str, width: int = 512, 
                           height: int = 512, background_color: str = "white") -> Tuple[torch.Tensor, torch.Tensor]:
        """将EPS转换为图像"""
        
        if not eps_file.strip():
            # 创建空白图像
            blank_image = self._create_blank_image(width, height)
            return blank_image, torch.ones((1, height, width), dtype=torch.float32)
        
        # 查找EPS文件
        eps_path = self._find_eps_file(eps_file)
        if eps_path is None:
            # 创建错误图像
            error_image = self._create_error_image(width, height, f"文件未找到: {eps_file}")
            return error_image, torch.ones((1, height, width), dtype=torch.float32)
        
        # 检查文件格式
        if eps_path.suffix.lower() != '.eps':
            error_image = self._create_error_image(width, height, "文件不是EPS格式")
            return error_image, torch.ones((1, height, width), dtype=torch.float32)
        
        # 转换为图像
        try:
            image = self._convert_eps_to_pil(eps_path, width, height, background_color)
            
            # 转换为tensor
            img_array = np.array(image).astype(np.float32) / 255.0
            
            # 分离RGB和Alpha
            if image.mode == 'RGBA':
                rgb_array = img_array[:, :, :3]
                alpha_array = img_array[:, :, 3]
            else:
                rgb_array = img_array
                alpha_array = np.ones((height, width), dtype=np.float32)
            
            # 转换为torch tensor
            rgb_tensor = torch.from_numpy(rgb_array)[None,]
            alpha_tensor = torch.from_numpy(alpha_array)[None,]
            
            return rgb_tensor, alpha_tensor
            
        except Exception as e:
            error_image = self._create_error_image(width, height, f"转换失败: {str(e)[:50]}")
            return error_image, torch.ones((1, height, width), dtype=torch.float32)
    
    def _find_eps_file(self, filename: str) -> Optional[Path]:
        """查找EPS文件"""
        # 检查是否为绝对路径
        eps_path = Path(filename)
        if eps_path.is_absolute() and eps_path.exists():
            return eps_path
        
        # 在输出目录中查找
        output_dir = Path(folder_paths.get_output_directory())
        
        # 检查各种可能的位置
        possible_paths = [
            output_dir / filename,
            output_dir / f"{filename}.eps",
            output_dir / "svg_to_eps" / filename,
            output_dir / "svg_to_eps" / f"{filename}.eps",
            output_dir / "batch_svg_to_eps" / "**" / filename,
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # 使用glob搜索
        for pattern in [f"**/{filename}", f"**/{filename}.eps"]:
            matches = list(output_dir.rglob(pattern))
            if matches:
                return matches[0]
        
        return None
    
    def _convert_eps_to_pil(self, eps_path: Path, width: int, height: int, 
                           bg_color: str) -> Image.Image:
        """将EPS转换为PIL图像"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # 使用Inkscape转换为PNG
            if SVGEPSConverterCore.check_inkscape():
                cmd = [
                    'inkscape',
                    '--export-type=png',
                    f'--export-filename={tmp_path}',
                    f'--export-width={width}',
                    f'--export-height={height}',
                    '--export-area-drawing',
                    str(eps_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, shell=False, timeout=30)
                
                if result.returncode != 0:
                    raise Exception(f"Inkscape转换失败: {result.stderr[:200]}")
            
            # 使用Ghostscript（如果Inkscape不可用）
            else:
                # 检查Ghostscript
                try:
                    result = subprocess.run(['gs', '--version'], 
                                          capture_output=True, text=True, shell=False, timeout=2)
                    gs_available = result.returncode == 0
                except:
                    gs_available = False
                
                if not gs_available:
                    raise Exception("需要Inkscape或Ghostscript来转换EPS文件")
                
                dpi = int(max(width, height) / 10 * 72)  # 估算DPI
                
                cmd = [
                    'gs',
                    '-dSAFER',
                    '-dBATCH',
                    '-dNOPAUSE',
                    '-dEPSCrop',
                    '-sDEVICE=png16m',
                    f'-r{dpi}',
                    f'-g{width}x{height}',
                    f'-sOutputFile={tmp_path}',
                    str(eps_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, shell=False, timeout=30)
                
                if result.returncode != 0:
                    raise Exception(f"Ghostscript转换失败: {result.stderr[:200]}")
            
            # 加载图像
            img = Image.open(tmp_path)
            
            # 处理背景
            if bg_color == "transparent" and img.mode != 'RGBA':
                img = img.convert('RGBA')
            elif img.mode == 'RGBA' and bg_color != "transparent":
                # 合成背景
                bg_color_rgb = (255, 255, 255) if bg_color == "white" else (0, 0, 0)
                bg = Image.new('RGB', img.size, bg_color_rgb)
                bg.paste(img, mask=img.split()[3])
                img = bg
            
            return img
            
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    def _create_blank_image(self, width: int, height: int) -> torch.Tensor:
        """创建空白图像"""
        array = np.ones((height, width, 3), dtype=np.float32) * 0.5  # 灰色
        return torch.from_numpy(array)[None,]
    
    def _create_error_image(self, width: int, height: int, message: str) -> torch.Tensor:
        """创建错误提示图像"""
        # 创建红色背景
        array = np.ones((height, width, 3), dtype=np.float32)
        array[:, :, 0] = 1.0  # 红色通道
        array[:, :, 1] = 0.8  # 绿色通道
        array[:, :, 2] = 0.8  # 蓝色通道
        
        return torch.from_numpy(array)[None,]


class CheckSVGDependenciesNode:
    """检查SVG转换依赖节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "check_dependencies"
    CATEGORY = "utils"
    OUTPUT_NODE = True
    
    def check_dependencies(self):
        """检查依赖"""
        deps = SVGEPSConverterCore.get_available_methods()
        
        lines = ["🔧 SVG到EPS转换依赖检查:", "=" * 50, ""]
        
        # Inkscape
        if deps["inkscape"]:
            lines.append("✅ Inkscape: 已安装 (推荐)")
            try:
                result = subprocess.run(['inkscape', '--version'], 
                                      capture_output=True, text=True, shell=False, timeout=2)
                version = result.stdout.split('\n')[0] if result.stdout else "未知版本"
                lines.append(f"   版本: {version}")
            except:
                lines.append("   版本: 无法获取")
        else:
            lines.append("❌ Inkscape: 未安装")
            lines.append("   安装指南:")
            lines.append("   - Linux: sudo apt install inkscape")
            lines.append("   - macOS: brew install inkscape")
            lines.append("   - Windows: 从 inkscape.org 下载")
        
        lines.append("")
        
        # CairoSVG
        if deps["cairosvg"]:
            lines.append("✅ CairoSVG: 已安装")
            try:
                import cairosvg
                lines.append(f"   版本: {cairosvg.__version__}")
            except:
                lines.append("   版本: 无法获取")
        else:
            lines.append("❌ CairoSVG: 未安装")
            lines.append("   安装: pip install cairosvg")
        
        lines.append("")
        
        # svglib
        if deps["svglib"]:
            lines.append("✅ svglib: 已安装")
            try:
                import svglib
                lines.append(f"   版本: {svglib.__version__}")
            except:
                lines.append("   版本: 无法获取")
        else:
            lines.append("❌ svglib: 未安装")
            lines.append("   安装: pip install svglib reportlab")
        
        lines.append("")
        lines.append("=" * 50)
        lines.append("")
        lines.append("📁 文件目录:")
        lines.append(f"   输入目录: {folder_paths.get_input_directory()}")
        lines.append(f"   输出目录: {folder_paths.get_output_directory()}")
        lines.append(f"   SVG转换输出: {folder_paths.get_output_directory()}/svg_to_eps/")
        lines.append(f"   批量转换输出: {folder_paths.get_output_directory()}/batch_svg_to_eps/")
        
        lines.append("")
        lines.append("💡 使用方法:")
        lines.append("   1. 将SVG文件放入输入目录")
        lines.append("   2. 使用SVGToEPS节点转换")
        lines.append("   3. 使用EPSToImage节点预览")
        
        info = "\n".join(lines)
        return (info,)


# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "SVGToEPS": SVGToEPSNode,
    "BatchSVGtoEPS": BatchSVGtoEPSNode,
    "EPSToImage": EPSToImageNode,
    "CheckSVGDependencies": CheckSVGDependenciesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVGToEPS": "🔄 SVG to EPS",
    "BatchSVGtoEPS": "📦 Batch SVG to EPS",
    "EPSToImage": "👁️ EPS to Image",
    "CheckSVGDependencies": "🔍 Check SVG Dependencies",
}

# 导出
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']