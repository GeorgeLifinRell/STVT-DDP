import sys
import yt_dlp

def download_video(url, output_path='.'):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'merge_output_format': 'mp4',
        # aria2c as external downloader with multiple connections
        'external_downloader': 'aria2c',
        'external_downloader_args': ['-x', '16', '-k', '1M']
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python yt_dlp_downloader.py <YouTube URL> [output_path]")
        sys.exit(1)

    video_url = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else '.'
    download_video(video_url, save_path)
