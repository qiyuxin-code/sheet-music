const { apiBase } = require('../../config');

function statusText(status) {
  const map = {
    pending: '待处理',
    doing: '处理中',
    ok: '成功',
    error: '失败',
  };
  return map[status] || status;
}

function uploadOne(path, name) {
  return new Promise((resolve, reject) => {
    wx.uploadFile({
      url: `${apiBase}/api/annotate`,
      filePath: path,
      name: 'file',
      formData: { filename: name },
      success(res) {
        if (res.statusCode !== 200) {
          reject(new Error(`服务器错误 ${res.statusCode}`));
          return;
        }
        try {
          const data = JSON.parse(res.data);
          if (data.detail) {
            reject(new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)));
            return;
          }
          resolve(data);
        } catch (e) {
          reject(new Error('响应解析失败'));
        }
      },
      fail(err) {
        reject(new Error(err.errMsg || '上传失败'));
      },
    });
  });
}

function writeResultImage(id, base64) {
  return new Promise((resolve, reject) => {
    const filePath = `${wx.env.USER_DATA_PATH}/jianpu_${id}.jpg`;
    wx.getFileSystemManager().writeFile({
      filePath, data: base64, encoding: 'base64',
      success: () => resolve(filePath),
      fail: () => reject(new Error('结果图片保存失败，请重试')),
    });
  });
}

Page({
  data: {
    apiBase,
    items: [],
    busy: false,
    progressHint: '',
    summary: '',
  },

  onChoose() {
    if (this.data.busy) return;
    wx.chooseMedia({
      count: 9,
      mediaType: ['image'],
      sourceType: ['album', 'camera'],
      success: (res) => {
        const picked = (res.tempFiles || []).map((f, i) => ({
          id: `${Date.now()}_${i}`,
          path: f.tempFilePath,
          name: f.tempFilePath.split('/').pop() || `image_${i + 1}.jpg`,
          status: 'pending',
          statusText: statusText('pending'),
          thumbUrl: '',
          resultPath: '',
          error: '',
        }));
        if (!picked.length) return;
        this.releaseResults();
        this.setData({ items: picked, summary: '' });
        this.runBatch();
      },
    });
  },

  async runBatch() {
    const items = this.data.items.slice();
    this.setData({ busy: true, progressHint: '准备上传…' });
    let ok = 0;
    let err = 0;
    for (let i = 0; i < items.length; i++) {
      items[i].status = 'doing';
      items[i].statusText = statusText('doing');
      this.setData({
        items,
        progressHint: `处理 ${i + 1} / ${items.length}`,
      });
      try {
        const data = await uploadOne(items[i].path, items[i].name);
        if (!data.img) throw new Error('未收到结果图片');
        items[i].resultPath = await writeResultImage(items[i].id, data.img);
        items[i].status = 'ok';
        items[i].statusText = statusText('ok');
        items[i].name = data.filename || items[i].name;
        items[i].thumbUrl = items[i].resultPath;
        ok += 1;
      } catch (e) {
        items[i].status = 'error';
        items[i].statusText = statusText('error');
        items[i].error = e.message || String(e);
        err += 1;
      }
      this.setData({ items });
    }
    this.setData({
      busy: false,
      progressHint: '',
      summary: `共 ${items.length} 张：成功 ${ok} 张，失败 ${err} 张`,
    });
  },

  onClear() {
    if (this.data.busy) return;
    this.releaseResults();
    this.setData({ items: [], summary: '' });
  },

  releaseResults() {
    const fs = wx.getFileSystemManager();
    this.data.items.forEach(item => {
      if (item.resultPath) fs.unlink({ filePath: item.resultPath, fail() {} });
    });
  },

  onUnload() { this.releaseResults(); },

  onPreview(e) {
    const id = e.currentTarget.dataset.id;
    const item = this.data.items.find((x) => x.id === id);
    if (!item || !item.resultPath) return;
    // 微信原生预览提供左右滑动、双指缩放和放大后的拖拽。
    const urls = this.data.items.filter(x => x.status === 'ok' && x.resultPath).map(x => x.resultPath);
    wx.previewImage({ urls, current: item.resultPath });
  },

  onSave(e) {
    const id = e.currentTarget.dataset.id;
    const item = this.data.items.find((x) => x.id === id);
    if (!item || !item.resultPath) return;
    wx.saveImageToPhotosAlbum({
      filePath: item.resultPath,
      success: () => wx.showToast({ title: '已保存', icon: 'success' }),
      fail: () => wx.showToast({ title: '请检查相册权限', icon: 'none' }),
    });
  },
});
