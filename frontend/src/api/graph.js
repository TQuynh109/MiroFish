import service, { requestWithRetry } from './index'

/**
 * Tạo ontology (upload file + yêu cầu mô phỏng)
 * @param {Object} data - Bao gồm files, simulation_requirement, project_name...
 * @returns {Promise}
 */
export function generateOntology(formData) {
  return requestWithRetry(() => 
    service({
      url: '/api/graph/ontology/generate',
      method: 'post',
      data: formData,
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  )
}

/**
 * Xây dựng graph
 * @param {Object} data - Bao gồm project_id, graph_name...
 * @returns {Promise}
 */
export function buildGraph(data) {
  return requestWithRetry(() =>
    service({
      url: '/api/graph/build',
      method: 'post',
      data
    })
  )
}

/**
 * Lấy trạng thái task
 * @param {String} taskId - ID của task
 * @returns {Promise}
 */
export function getTaskStatus(taskId) {
  return service({
    url: `/api/graph/task/${taskId}`,
    method: 'get'
  })
}

/**
 * Lấy dữ liệu graph
 * @param {String} graphId - ID của graph
 * @returns {Promise}
 */
export function getGraphData(graphId) {
  return service({
    url: `/api/graph/data/${graphId}`,
    method: 'get'
  })
}

/**
 * Lấy thông tin project
 * @param {String} projectId - ID của project
 * @returns {Promise}
 */
export function getProject(projectId) {
  return service({
    url: `/api/graph/project/${projectId}`,
    method: 'get'
  })
}
