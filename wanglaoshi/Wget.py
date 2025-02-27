import requests
import os
import sys
from urllib.parse import urlparse
from tqdm import tqdm


class Wget:
    def __init__(self, url, output_dir='.', filename=None):
        self.url = url
        self.output_dir = output_dir
        self.filename = filename or self._get_filename_from_url()

    def _get_filename_from_url(self):
        """从 URL 中获取文件名"""
        parsed = urlparse(self.url)
        return os.path.basename(parsed.path) or 'downloaded_file'

    def _create_output_dir(self):
        """创建输出目录"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def download(self):
        """下载文件"""
        try:
            # 发送 HEAD 请求获取文件大小
            response = requests.head(self.url)
            total_size = int(response.headers.get('content-length', 0))

            # 创建输出目录
            self._create_output_dir()
            output_path = os.path.join(self.output_dir, self.filename)

            # 发送 GET 请求下载文件
            response = requests.get(self.url, stream=True)
            response.raise_for_status()  # 检查响应状态

            # 使用 tqdm 显示进度条
            with open(output_path, 'wb') as f, tqdm(
                    desc=self.filename,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)

            print(f"\n下载完成: {output_path}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"下载错误: {str(e)}")
            return False
        except Exception as e:
            print(f"发生错误: {str(e)}")
            return False


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python wget.py <URL> [输出目录] [文件名]")
        sys.exit(1)

    url = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    filename = sys.argv[3] if len(sys.argv) > 3 else None

    downloader = Wget(url, output_dir, filename)
    downloader.download()


if __name__ == "__main__":
    main()