/* 禾眼 HeYan Service Worker
 *
 * 文档 3.1 节要求"全离线可用"，所以这个 SW 的唯一职责是：
 * 把界面外壳（HTML/CSS/JS/图标/清单）在安装时预缓存，之后断网也能打开。
 *
 * 刻意不做的事：
 *   - 不缓存 /api/*。识别、语音、记录都必须打到本机服务进程，
 *     缓存旧结果会让农户照着过期诊断打药，这是安全问题不是性能问题。
 *   - 不缓存模型文件。onnxruntime 在服务端读磁盘，浏览器碰不到。
 */

const VERSION = 'heyan-v1.0.0';
const SHELL = `shell-${VERSION}`;
const RUNTIME = `runtime-${VERSION}`;

// 应用外壳：全部本机资源，无一个外链
const PRECACHE = [
  '/',
  '/index.html',
  '/static/style.css',
  '/static/app.js',
  '/manifest.webmanifest',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
  '/icons/maskable-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(SHELL)
      .then((cache) => cache.addAll(PRECACHE).catch(() => {
        // 单个资源缺失（比如图标还没生成）不该让整个安装失败，
        // 逐个放，能存多少存多少。
        return Promise.all(PRECACHE.map((url) => cache.add(url).catch(() => null)));
      }))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k !== SHELL && k !== RUNTIME).map((k) => caches.delete(k)),
      ))
      .then(() => self.clients.claim()),
  );
});

function isApi(url) {
  return url.pathname.startsWith('/api/');
}

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;   // 跨域一律不管
  if (isApi(url)) return;                            // 接口永远走网络

  // 页面导航：网络优先，断网回落到缓存的外壳，保证"打开就能用"
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((resp) => {
          const copy = resp.clone();
          caches.open(SHELL).then((c) => c.put('/index.html', copy)).catch(() => {});
          return resp;
        })
        .catch(() => caches.match('/index.html').then((hit) => hit || Response.error())),
    );
    return;
  }

  // 静态资源：缓存优先，后台补新（stale-while-revalidate）
  event.respondWith(
    caches.match(request).then((hit) => {
      const refresh = fetch(request)
        .then((resp) => {
          if (resp && resp.status === 200) {
            const copy = resp.clone();
            caches.open(RUNTIME).then((c) => c.put(request, copy)).catch(() => {});
          }
          return resp;
        })
        .catch(() => null);
      return hit || refresh || Response.error();
    }),
  );
});

// 界面里的"立即更新"按钮：跳过等待，让新版本外壳立刻生效
self.addEventListener('message', (event) => {
  if (event.data === 'skip-waiting') self.skipWaiting();
});
