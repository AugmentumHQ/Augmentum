/**
 * Shared YouTube IFrame API loader.
 *
 * Loads the YouTube IFrame API script exactly once.
 * Multiple callers get the same promise.
 *
 * Usage:
 *   import { loadYouTubeAPI } from './yt-api.js';
 *   const YT = await loadYouTubeAPI();
 *   const player = new YT.Player('el', { videoId: '...' });
 */

let _promise = null;

export function loadYouTubeAPI() {
  if (_promise) return _promise;

  // Already loaded (e.g. by another script path)
  if (window.YT && window.YT.Player) {
    _promise = Promise.resolve(window.YT);
    return _promise;
  }

  _promise = new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      _promise = null;
      reject(new Error('YouTube API load timeout'));
    }, 15000);

    window.onYouTubeIframeAPIReady = () => {
      clearTimeout(timeout);
      resolve(window.YT);
    };

    const tag = document.createElement('script');
    tag.src = 'https://www.youtube.com/iframe_api';
    tag.onerror = () => {
      clearTimeout(timeout);
      _promise = null;
      reject(new Error('YouTube API script failed to load'));
    };
    document.head.appendChild(tag);
  });

  return _promise;
}
