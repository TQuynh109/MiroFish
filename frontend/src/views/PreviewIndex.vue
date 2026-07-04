<template>
  <div class="preview-index">
    <header class="ph-header">
      <div class="ph-brand">SIMPETRO · PREVIEW</div>
    </header>

    <!-- Chọn project -->
    <div class="ph-picker">
      <div class="ph-picker-row">
        <label class="ph-label">Project / Simulation:</label>
        <select v-model="selectedId" class="ph-select" :disabled="loading">
          <option :value="null">— ID giả (demo, backend trả not-found) —</option>
          <option v-for="p in projects" :key="p.simulation_id" :value="p.simulation_id">
            {{ shortId(p.simulation_id) }} · {{ title(p) }}
          </option>
        </select>
        <button class="ph-reload" @click="loadProjects" :disabled="loading">
          {{ loading ? '...' : '↻ Tải lại' }}
        </button>
      </div>
      <div class="ph-picker-status">
        <template v-if="loading">Đang tải lịch sử project từ backend...</template>
        <template v-else-if="loadError">⚠️ Không tải được lịch sử ({{ loadError }}). Backend (localhost:5002) có đang chạy không?</template>
        <template v-else-if="projects.length === 0">Không có project nào trong lịch sử. Dùng ID giả để review layout.</template>
        <template v-else-if="selected">
          Đang dùng: project_id=<b>{{ selected.project_id || '∅' }}</b> · report_id=<b>{{ selected.report_id || '∅' }}</b>
          <span v-if="!selected.project_id" class="ph-miss"> · ⚠ chưa có project_id → Step 1 sẽ rỗng</span>
          <span v-if="!selected.report_id" class="ph-miss"> · ⚠ chưa có report_id → Step 4/5 sẽ rỗng</span>
        </template>
        <template v-else>Đang dùng ID giả — chỉ review được layout, panel phải sẽ rỗng.</template>
      </div>
    </div>

    <div class="ph-mode" :class="{ 'ph-mode-safe': previewMode, 'ph-mode-live': !previewMode }">
      <div class="ph-mode-row">
        <span class="ph-mode-icon">{{ previewMode ? '👁' : '⚡' }}</span>
        <div class="ph-mode-text">
          <span class="ph-mode-label">Chế độ vận hành</span>
          <span class="ph-mode-desc">Chọn phương thức xử lý dữ liệu và mô phỏng cho Bước 2 và Bước 3</span>
        </div>
        <select v-model="previewMode" class="ph-mode-select">
          <option :value="true">👁 View — chỉ xem dữ liệu cũ</option>
          <option :value="false">⚡ Live — chạy thật</option>
        </select>
      </div>
      <div class="ph-mode-status">
        <template v-if="previewMode">
          Đang ở chế độ <b>View</b>: mở Step 2 chỉ load cấu hình đã chuẩn bị, mở Step 3 chỉ xem kết quả mô phỏng cũ.
        </template>
        <template v-else>
          Đang ở chế độ <b>Live</b>: mở Step 2 sẽ gọi <code>prepareSimulation</code> (sinh agent, tốn LLM),
          mở Step 3 sẽ <b>tự chạy mô phỏng thật</b>.
        </template>
      </div>
    </div>

    <ul class="ph-list">
      <li v-for="page in pages" :key="page.key" class="ph-item">
        <router-link
          :to="page.path"
          class="ph-link"
          :class="{ 'is-write': page.write, 'is-disabled': page.disabled }"
          @click="page.disabled && $event.preventDefault()"
        >
          <span class="ph-step">{{ page.step }}</span>
          <span class="ph-body">
            <span class="ph-name">
              {{ page.name }}
              <span v-if="page.write" class="ph-badge">GHI DỮ LIỆU</span>
              <span v-if="page.disabled" class="ph-badge ph-badge-off">thiếu id</span>
            </span>
            <span class="ph-file">{{ page.file }}</span>
          </span>
          <span class="ph-arrow">→</span>
        </router-link>
      </li>
    </ul>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { getSimulationHistory } from '../api/simulation'

// ID giả mặc định khi chưa chọn project thật
const FAKE = { project_id: 'demo-project', simulation_id: 'demo-sim', report_id: 'demo-report' }

const projects = ref([])
const loading = ref(false)
const loadError = ref('')
const selectedId = ref(null)
// Bật: Step 2/3 chỉ load dữ liệu cũ (?preview=1), không gọi LLM / không chạy lại mô phỏng
const previewMode = ref(true)

const selected = computed(() =>
  projects.value.find(p => p.simulation_id === selectedId.value) || null
)

// id thực tế dùng để dựng link (project thật nếu có, ngược lại dùng FAKE)
const ids = computed(() => ({
  project_id: selected.value?.project_id || FAKE.project_id,
  simulation_id: selected.value?.simulation_id || FAKE.simulation_id,
  report_id: selected.value?.report_id || FAKE.report_id,
}))

// Khi chọn project thật, disable trang nào thiếu id tương ứng để tránh mở ra rỗng/nhầm
const hasReal = computed(() => !!selected.value)
const missingProject = computed(() => hasReal.value && !selected.value.project_id)
const missingReport = computed(() => hasReal.value && !selected.value.report_id)

// Query dùng chung để PreviewNav ở các trang con biết id của 3 loại tài nguyên
// và biết đang ở chế độ preview để hiện thanh điều hướng prev/next
const navQuery = computed(() => {
  const q = new URLSearchParams()
  q.set('preview', '1')
  q.set('projectId', ids.value.project_id)
  q.set('simulationId', ids.value.simulation_id)
  q.set('reportId', ids.value.report_id)
  return q.toString()
})

const pages = computed(() => [
  { key: 'home', step: '—', name: 'Home / Trang chủ', file: 'Home.vue', path: '/', write: false, disabled: false },
  { key: 's1', step: 'Step 1', name: 'Build Graph', file: 'MainView.vue',
    path: `/process/${ids.value.project_id}?${navQuery.value}`, write: false, disabled: missingProject.value },
  { key: 's2', step: 'Step 2', name: 'Environment Setup', file: 'SimulationView.vue',
    path: `/simulation/${ids.value.simulation_id}?${navQuery.value}`,
    write: !previewMode.value, disabled: false },
  { key: 's3', step: 'Step 3', name: 'Start Simulation', file: 'SimulationRunView.vue',
    path: `/simulation/${ids.value.simulation_id}/start?${navQuery.value}`,
    write: !previewMode.value, disabled: false },
  { key: 's4', step: 'Step 4', name: 'Report Generation', file: 'ReportView.vue',
    path: `/report/${ids.value.report_id}?${navQuery.value}`, write: false, disabled: missingReport.value },
  { key: 's5', step: 'Step 5', name: 'Deep Interaction', file: 'InteractionView.vue',
    path: `/interaction/${ids.value.report_id}?${navQuery.value}`, write: false, disabled: missingReport.value },
])

const shortId = (id) => (id ? id.substring(0, 8) : '?')
const title = (p) => {
  const t = p.project_name || p.simulation_requirement || '(no requirement)'
  return t.length > 40 ? t.substring(0, 40) + '…' : t
}

const loadProjects = async () => {
  loading.value = true
  loadError.value = ''
  try {
    const res = await getSimulationHistory(50)
    if (res.success) {
      projects.value = res.data || []
    } else {
      loadError.value = res.error || 'unknown'
    }
  } catch (e) {
    loadError.value = e.message || 'request failed'
  } finally {
    loading.value = false
  }
}

onMounted(loadProjects)
</script>

<style scoped>
.preview-index {
  min-height: 100vh; background: #fff; color: #000;
  font-family: 'JetBrains Mono', monospace; padding: 40px; max-width: 960px; margin: 0 auto;
}
.ph-header { margin-bottom: 20px; }
.ph-brand { font-weight: 800; font-size: 1.4rem; letter-spacing: 1px; }
.ph-note { color: #666; font-size: 0.85rem; margin-top: 6px; }

.ph-picker { border: 1px solid #E0E0E0; padding: 16px 20px; margin-bottom: 20px; background: #FAFAFA; }
.ph-picker-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.ph-label { font-size: 0.8rem; font-weight: 700; }
.ph-select { flex: 1; min-width: 280px; padding: 8px 10px; font-family: inherit; font-size: 0.82rem; border: 1px solid #ccc; background: #fff; }
.ph-reload { padding: 8px 14px; font-family: inherit; font-size: 0.78rem; border: 1px solid #000; background: #000; color: #fff; cursor: pointer; }
.ph-reload:disabled { opacity: 0.5; cursor: default; }
.ph-picker-status { margin-top: 10px; font-size: 0.78rem; color: #555; line-height: 1.5; }
.ph-miss { color: #B23A1A; }

.ph-mode {
  border: 1px solid #FFD9CC; background: #FFF5F2; color: #B23A1A;
  padding: 14px 18px; margin-bottom: 24px; transition: background 0.15s, border-color 0.15s;
}
.ph-mode.ph-mode-safe { background: #F2FBF4; border-color: #BFE8C8; color: #1B6B33; }

.ph-mode-row { display: flex; align-items: center; gap: 14px; }
.ph-mode-icon { font-size: 1.3rem; flex-shrink: 0; }
.ph-mode-text { flex: 1; display: flex; flex-direction: column; gap: 2px; min-width: 200px; }
.ph-mode-label { font-weight: 700; font-size: 0.85rem; }
.ph-mode-desc { font-size: 0.72rem; opacity: 0.8; }

.ph-mode-select {
  font-family: inherit; font-size: 0.78rem; font-weight: 600; padding: 8px 12px;
  border: 1px solid currentColor; background: #fff; color: inherit; cursor: pointer; min-width: 220px;
}

.ph-mode-status { margin-top: 10px; padding-top: 10px; border-top: 1px dashed currentColor; font-size: 0.78rem; line-height: 1.6; opacity: 0.9; }
.ph-mode-status code { background: #fff; padding: 1px 5px; border: 1px solid currentColor; }

.ph-list { list-style: none; display: flex; flex-direction: column; gap: 10px; }
.ph-link {
  display: flex; align-items: center; gap: 16px; padding: 16px 20px;
  border: 1px solid #E0E0E0; text-decoration: none; color: #000; transition: all 0.15s;
}
.ph-link:hover { border-color: #FF6B35; background: #FFF9F6; transform: translateX(4px); }
.ph-link.is-write { border-color: #E0A0A0; background: #FFF8F8; }
.ph-link.is-write:hover { border-color: #D32F2F; background: #FFF0F0; }
.ph-link.is-disabled { opacity: 0.4; pointer-events: none; }

.ph-step { flex-shrink: 0; width: 64px; font-weight: 700; font-size: 0.75rem; color: #FF6B35; letter-spacing: 0.05em; }
.ph-body { flex: 1; display: flex; flex-direction: column; gap: 3px; }
.ph-name { font-weight: 600; font-size: 0.95rem; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.ph-file { font-size: 0.72rem; color: #999; }
.ph-path { font-size: 0.78rem; color: #666; }
.ph-arrow { color: #FF6B35; font-size: 1.1rem; }

.ph-badge { font-size: 0.6rem; font-weight: 700; background: #D32F2F; color: #fff; padding: 2px 6px; letter-spacing: 0.05em; }
.ph-badge-off { background: #999; }
</style>
