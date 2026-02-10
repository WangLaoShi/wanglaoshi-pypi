"""
网络文件下载模块 (WebGetter.py)

这个模块提供了一个简单的文件下载工具类，类似于Linux的wget命令。
主要功能：
1. 从URL下载文件
2. 显示下载进度条
3. 自动创建输出目录
4. 支持自定义文件名

使用场景：
- 下载字体文件（如SimHei.ttf）
- 下载数据集文件
- 下载其他资源文件

主要类：
- Wget: 文件下载工具类
"""

import requests  # HTTP请求库，用于下载文件
import os        # 操作系统接口，用于文件路径操作
import sys       # 系统相关（虽然导入但未使用，可能是为了兼容性）
from urllib.parse import urlparse  # URL解析工具，用于从URL提取文件名
from tqdm import tqdm  # 进度条库，用于显示下载进度


class Wget:
    """
    文件下载工具类（类似Linux的wget命令）
    
    这个类提供了从网络下载文件的功能，包括：
    - 自动从URL提取文件名
    - 显示下载进度条
    - 自动创建输出目录
    - 错误处理
    
    使用示例:
        downloader = Wget(
            url='https://example.com/file.zip',
            output_dir='./downloads',
            filename='my_file.zip'
        )
        downloader.download()
    """
    
    def __init__(self, url, output_dir='.', filename=None):
        """
        初始化下载器
        
        参数:
            url: 要下载的文件URL（完整地址）
                例如: "https://example.com/file.zip"
            output_dir: 文件保存的目录，默认为当前目录 '.'
            filename: 保存的文件名，如果为None则从URL自动提取
        """
        self.url = url  # 保存下载地址
        self.output_dir = output_dir  # 保存输出目录
        # 如果未指定文件名，从URL中自动提取
        self.filename = filename or self._get_filename_from_url()

    def _get_filename_from_url(self):
        """
        从URL中自动提取文件名
        
        这个方法会解析URL，提取路径中的文件名部分。
        如果无法提取，则使用默认文件名。
        
        返回:
            文件名字符串
            例如: "file.zip" 或 "downloaded_file"（如果无法提取）
            
        示例:
            URL: "https://example.com/path/to/file.zip"
            返回: "file.zip"
        """
        # urlparse() 解析URL，返回包含各个部分的对象
        parsed = urlparse(self.url)
        # os.path.basename() 从路径中提取文件名
        # 例如: "/path/to/file.zip" -> "file.zip"
        # 如果路径为空或无法提取，使用默认文件名
        return os.path.basename(parsed.path) or 'downloaded_file'

    def _create_output_dir(self):
        """
        创建输出目录（如果不存在）
        
        这个方法会检查输出目录是否存在，如果不存在则创建它。
        这样可以确保下载文件时目录已经准备好。
        """
        # 检查目录是否存在
        if not os.path.exists(self.output_dir):
            # 如果不存在，创建目录（包括所有必要的父目录）
            os.makedirs(self.output_dir)

    def download(self):
        """
        执行文件下载
        
        这个方法会：
        1. 获取文件大小（用于显示进度条）
        2. 创建输出目录
        3. 下载文件并显示进度条
        4. 保存文件到指定位置
        
        返回:
            True: 下载成功
            False: 下载失败
            
        异常处理:
            - 网络错误：捕获requests异常
            - 其他错误：捕获所有异常并返回False
        """
        try:
            # ========== 步骤1：获取文件大小 ==========
            # 发送HEAD请求（只获取响应头，不下载文件内容）
            # 这样可以快速获取文件大小，而不需要下载整个文件
            response = requests.head(self.url)
            # 从响应头中获取文件大小（content-length字段）
            # 如果无法获取，默认为0
            total_size = int(response.headers.get('content-length', 0))

            # ========== 步骤2：准备输出路径 ==========
            # 创建输出目录（如果不存在）
            self._create_output_dir()
            # 构建完整的文件保存路径
            output_path = os.path.join(self.output_dir, self.filename)

            # ========== 步骤3：下载文件 ==========
            # 发送GET请求下载文件
            # stream=True 表示以流式方式下载（分块下载，节省内存）
            response = requests.get(self.url, stream=True)
            # 检查HTTP响应状态码，如果不是200会抛出异常
            response.raise_for_status()

            # ========== 步骤4：保存文件并显示进度 ==========
            # 以二进制写入模式打开文件
            with open(output_path, 'wb') as f, tqdm(
                    desc=self.filename,      # 进度条描述（显示文件名）
                    total=total_size,       # 总大小（用于计算进度百分比）
                    unit='iB',              # 单位：字节（iB表示二进制字节）
                    unit_scale=True,        # 自动缩放单位（KB, MB, GB）
                    unit_divisor=1024,      # 缩放除数（1024字节=1KB）
            ) as pbar:
                # 分块读取和写入文件
                # iter_content() 每次返回指定大小的数据块
                # chunk_size=1024 表示每次读取1024字节（1KB）
                for data in response.iter_content(chunk_size=1024):
                    # 将数据块写入文件
                    size = f.write(data)
                    # 更新进度条（增加已下载的字节数）
                    pbar.update(size)

            # 下载完成提示
            print(f"\n下载完成: {output_path}")
            return True

        except requests.exceptions.RequestException as e:
            # 捕获网络请求相关的异常（连接错误、超时等）
            print(f"下载错误: {str(e)}")
            return False
        except Exception as e:
            # 捕获其他所有异常（文件写入错误等）
            print(f"发生错误: {str(e)}")
            return False


# def main():
#     """主函数"""
#     if len(sys.argv) < 2:
#         print("使用方法: python wget.py <URL> [输出目录] [文件名]")
#         sys.exit(1)
#
#     url = sys.argv[1]
#     output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
#     filename = sys.argv[3] if len(sys.argv) > 3 else None
#
#     downloader = Wget(url, output_dir, filename)
#     downloader.download()
#
#
# if __name__ == "__main__":
#     main()