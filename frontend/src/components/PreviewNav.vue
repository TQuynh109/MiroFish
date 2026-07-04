<template>
  <div v-if="isPreview && !inline" class="preview-nav">
    <router-link to="/preview" class="pn-home" title="Về danh sách preview">⌂</router-link>
    <button class="pn-btn" :disabled="!prevPage" @click="go(prevPage)">← {{ prevPage?.name || 'Previous' }}</button>
    <span class="pn-current">{{ currentPage?.step || '' }}</span>
    <button class="pn-btn" :disabled="!nextPage" @click="go(nextPage)">{{ nextPage?.name || 'Next' }} →</button>
  </div>
  <template v-else-if="isPreview && inline">
    <button v-if="side === 'prev'" class="pn-btn-inline" :disabled="!prevPage" @click="go(prevPage)">← {{ prevPage?.name || 'Previous' }}</button>
    <button v-else-if="side === 'next'" class="pn-btn-inline" :disabled="!nextPage" @click="go(nextPage)">{{ nextPage?.name || 'Next' }} →</button>
  </template>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const props = defineProps({
  inline: { type: Boolean, default: false },
  side: { type: String, default: '' } // 'prev' | 'next', chỉ dùng khi inline
})

const route = useRoute()
const router = useRouter()

// Chỉ hiện thanh nav khi đến từ trang preview (?preview=1 hoặc còn id trong query)
const isPreview = computed(() => {
  return route.query.preview === '1' || !!(route.query.projectId || route.query.simulationId || route.query.reportId)
})

// Id ngữ cảnh được truyền xuyên suốt 5 bước qua query string
const ids = computed(() => ({
  projectId: route.query.projectId || '',
  simulationId: route.query.simulationId || '',
  reportId: route.query.reportId || '',
}))

const withQuery = (path) => {
  const q = new URLSearchParams()
  q.set('preview', '1')
  if (ids.value.projectId) q.set('projectId', ids.value.projectId)
  if (ids.value.simulationId) q.set('simulationId', ids.value.simulationId)
  if (ids.value.reportId) q.set('reportId', ids.value.reportId)
  return `${path}?${q.toString()}`
}

const steps = computed(() => [
  { key: 's1', routeName: 'Process', step: 'Step 1', name: 'Build Graph', path: ids.value.projectId ? `/process/${ids.value.projectId}` : null },
  { key: 's2', routeName: 'Simulation', step: 'Step 2', name: 'Environment Setup', path: ids.value.simulationId ? `/simulation/${ids.value.simulationId}` : null },
  { key: 's3', routeName: 'SimulationRun', step: 'Step 3', name: 'Start Simulation', path: ids.value.simulationId ? `/simulation/${ids.value.simulationId}/start` : null },
  { key: 's4', routeName: 'Report', step: 'Step 4', name: 'Report Generation', path: ids.value.reportId ? `/report/${ids.value.reportId}` : null },
  { key: 's5', routeName: 'Interaction', step: 'Step 5', name: 'Deep Interaction', path: ids.value.reportId ? `/interaction/${ids.value.reportId}` : null },
])

const currentIndex = computed(() => steps.value.findIndex(s => s.routeName === route.name))
const currentPage = computed(() => currentIndex.value >= 0 ? steps.value[currentIndex.value] : null)
const prevPage = computed(() => {
  for (let i = currentIndex.value - 1; i >= 0; i--) {
    if (steps.value[i].path) return steps.value[i]
  }
  return null
})
const nextPage = computed(() => {
  for (let i = currentIndex.value + 1; i < steps.value.length; i++) {
    if (steps.value[i].path) return steps.value[i]
  }
  return null
})

const go = (page) => {
  if (page?.path) router.push(withQuery(page.path))
}
</script>

<style scoped>
.preview-nav {
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 10px;
  background: #000;
  color: #fff;
  padding: 8px 10px;
  border-radius: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  z-index: 9999;
  box-shadow: 0 4px 16px rgba(0,0,0,0.25);
}
.pn-home {
  color: #fff;
  text-decoration: none;
  padding: 6px 8px;
  border-radius: 6px;
  opacity: 0.8;
}
.pn-home:hover { opacity: 1; background: rgba(255,255,255,0.1); }
.pn-btn {
  background: #fff;
  color: #000;
  border: none;
  padding: 6px 12px;
  border-radius: 6px;
  font-family: inherit;
  font-size: inherit;
  font-weight: 600;
  cursor: pointer;
}
.pn-btn:disabled { opacity: 0.3; cursor: default; }
.pn-btn:not(:disabled):hover { background: #FF6B35; color: #fff; }
.pn-current { color: #999; padding: 0 4px; }

.pn-btn-inline {
  background: #000;
  color: #fff;
  border: none;
  padding: 6px 14px;
  border-radius: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}
.pn-btn-inline:disabled { opacity: 0.3; cursor: default; }
.pn-btn-inline:not(:disabled):hover { background: #FF6B35; color: #fff; }
</style>
