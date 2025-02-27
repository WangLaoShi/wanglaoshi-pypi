from wanglaoshi import WebGetter

downloader = WebGetter.Wget(url='https://gitee.com/lincoln/fonts/raw/master/SimHei.ttf',
        output_dir='.',
        filename='simhei.ttf',
    )
downloader.download()