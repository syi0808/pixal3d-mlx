import {copyFile, mkdir} from 'node:fs/promises';

const destination = new URL('../pixalmlx/server/static/vendor/', import.meta.url);
await mkdir(destination, {recursive: true});
await copyFile(new URL('node_modules/@google/model-viewer/dist/model-viewer.min.js', import.meta.url), new URL('model-viewer.min.js', destination));
await copyFile(new URL('node_modules/@google/model-viewer/LICENSE', import.meta.url), new URL('model-viewer.LICENSE', destination));
console.log('Self-hosted model-viewer assets are ready.');
