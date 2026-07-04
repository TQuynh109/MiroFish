<template>
  <div class="simulation-panel">
    <!-- Top Control Bar -->
    <div class="control-bar">
      <div class="timeline-stats" v-if="allActions.length > 0">
        <span class="total-count">TOTAL EVENTS: <span class="mono">{{ allActions.length }}</span></span>
        <span class="platform-breakdown">
          <span class="breakdown-item twitter">
            <svg class="mini-icon" viewBox="0 0 24 24" width="12" height="12" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"></path></svg>
            <span class="mono">{{ twitterActionsCount }}</span>
          </span>
          <span class="breakdown-divider">/</span>
          <span class="breakdown-item reddit">
            <svg class="mini-icon" viewBox="0 0 24 24" width="12" height="12" fill="currentColor"><path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.82 12.885c.014.109.014.219.014.328 0 3.36-3.51 6.087-7.834 6.087-4.324 0-7.834-2.727-7.834-6.087 0-.109 0-.219.014-.328a2.213 2.213 0 0 1-.71-1.643 2.221 2.221 0 0 1 2.222-2.222c.598 0 1.135.24 1.531.626 1.516-1.09 3.617-1.79 5.953-1.867l1.135-5.35a.161.161 0 0 1 .188-.128l3.762.792a1.484 1.484 0 1 1-.153.751l-3.417-.716-.996 4.699c2.302.096 4.363.79 5.859 1.867a2.16 2.16 0 0 1 1.531-.626 2.221 2.221 0 0 1 2.222 2.222c0 .724-.373 1.36-.938 1.735zM7.5 12.5c-.69 0-1.25.56-1.25 1.25s.56 1.25 1.25 1.25 1.25-.56 1.25-1.25-.56-1.25-1.25-1.25zm9.4 4.2c-.9.9-2.4 1.35-4.9 1.35-2.5 0-4-.45-4.9-1.35a.4.4 0 1 1 .566-.566c.71.71 1.9 1.116 4.334 1.116 2.434 0 3.624-.406 4.334-1.116a.4.4 0 1 1 .566.566zM16.5 15c-.69 0-1.25-.56-1.25-1.25s.56-1.25 1.25-1.25 1.25.56 1.25 1.25-.56 1.25-1.25 1.25z"></path></svg>
            <span class="mono">{{ redditActionsCount }}</span>
          </span>
        </span>
        <select v-model="actionTypeFilter" class="action-type-filter">
          <option value="ALL">All Actions</option>
          <option v-for="t in availableActionTypes" :key="t" :value="t">{{ t }}</option>
        </select>
      </div>

      <div class="action-controls">
        <button
          class="action-btn primary"
          :disabled="phase !== 2 || isGeneratingReport"
          @click="handleNextStep"
        >
          <span v-if="isGeneratingReport" class="loading-spinner-small"></span>
          {{ isGeneratingReport ? 'Starting...' : 'Start Generating Report' }}
          <span v-if="!isGeneratingReport" class="arrow-icon">→</span>
        </button>
      </div>
    </div>

    <!-- Main Content: Dual Timeline -->
    <div class="main-content-area" ref="scrollContainer">
      <!-- Timeline Header: Twitter/Reddit platform progress -->
      <div class="timeline-header">
        <div class="status-group">
          <!-- Twitter platform progress -->
          <div class="platform-status twitter" :class="{ active: runStatus.twitter_running, completed: runStatus.twitter_completed }">
            <div class="platform-header">
              <svg class="platform-icon" viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
                <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"></path>
              </svg>
              <span class="platform-name">Twitter</span>
              <span v-if="runStatus.twitter_completed" class="status-badge">
                <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="3">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
              </span>
            </div>
            <div class="platform-stats">
              <span class="stat">
                <span class="stat-label">ROUND</span>
                <span class="stat-value mono">{{ runStatus.twitter_current_round || 0 }}<span class="stat-total">/{{ runStatus.total_rounds || maxRounds || '-' }}</span></span>
              </span>
              <span class="stat">
                <span class="stat-label">Elapsed Time</span>
                <span class="stat-value mono">{{ twitterElapsedTime }}</span>
              </span>
              <span class="stat">
                <span class="stat-label">ACTS</span>
                <span class="stat-value mono">{{ runStatus.twitter_actions_count || 0 }}</span>
              </span>
            </div>
            <!-- Gợi ý các hành động khả dụng -->
            <div class="actions-tooltip">
              <div class="tooltip-title">Available Actions</div>
              <div class="tooltip-actions">
                <span class="tooltip-action">CREATE_POST</span>
                <span class="tooltip-action">LIKE_POST</span>
                <span class="tooltip-action">REPOST</span>
                <span class="tooltip-action">QUOTE_POST</span>
                <span class="tooltip-action">FOLLOW</span>
                <span class="tooltip-action">DO_NOTHING</span>
              </div>
            </div>
          </div>

          <!-- Reddit platform progress -->
          <div class="platform-status reddit" :class="{ active: runStatus.reddit_running, completed: runStatus.reddit_completed }">
            <div class="platform-header">
              <svg class="platform-icon" viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
                <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.82 12.885c.014.109.014.219.014.328 0 3.36-3.51 6.087-7.834 6.087-4.324 0-7.834-2.727-7.834-6.087 0-.109 0-.219.014-.328a2.213 2.213 0 0 1-.71-1.643 2.221 2.221 0 0 1 2.222-2.222c.598 0 1.135.24 1.531.626 1.516-1.09 3.617-1.79 5.953-1.867l1.135-5.35a.161.161 0 0 1 .188-.128l3.762.792a1.484 1.484 0 1 1-.153.751l-3.417-.716-.996 4.699c2.302.096 4.363.79 5.859 1.867a2.16 2.16 0 0 1 1.531-.626 2.221 2.221 0 0 1 2.222 2.222c0 .724-.373 1.36-.938 1.735zM7.5 12.5c-.69 0-1.25.56-1.25 1.25s.56 1.25 1.25 1.25 1.25-.56 1.25-1.25-.56-1.25-1.25-1.25zm9.4 4.2c-.9.9-2.4 1.35-4.9 1.35-2.5 0-4-.45-4.9-1.35a.4.4 0 1 1 .566-.566c.71.71 1.9 1.116 4.334 1.116 2.434 0 3.624-.406 4.334-1.116a.4.4 0 1 1 .566.566zM16.5 15c-.69 0-1.25-.56-1.25-1.25s.56-1.25 1.25-1.25 1.25.56 1.25 1.25-.56 1.25-1.25 1.25z"></path>
              </svg>
              <span class="platform-name">Reddit</span>
              <span v-if="runStatus.reddit_completed" class="status-badge">
                <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="3">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
              </span>
            </div>
            <div class="platform-stats">
              <span class="stat">
                <span class="stat-label">ROUND</span>
                <span class="stat-value mono">{{ runStatus.reddit_current_round || 0 }}<span class="stat-total">/{{ runStatus.total_rounds || maxRounds || '-' }}</span></span>
              </span>
              <span class="stat">
                <span class="stat-label">Elapsed Time</span>
                <span class="stat-value mono">{{ redditElapsedTime }}</span>
              </span>
              <span class="stat">
                <span class="stat-label">ACTS</span>
                <span class="stat-value mono">{{ runStatus.reddit_actions_count || 0 }}</span>
              </span>
            </div>
            <!-- Gợi ý các hành động khả dụng -->
            <div class="actions-tooltip">
              <div class="tooltip-title">Available Actions</div>
              <div class="tooltip-actions">
                <span class="tooltip-action">LIKE_POST</span>
                <span class="tooltip-action">DISLIKE_POST</span>
                <span class="tooltip-action">CREATE_POST</span>
                <span class="tooltip-action">CREATE_COMMENT</span>
                <span class="tooltip-action">LIKE_COMMENT</span>
                <span class="tooltip-action">DISLIKE_COMMENT</span>
                <span class="tooltip-action">SEARCH_POSTS</span>
                <span class="tooltip-action">SEARCH_USER</span>
                <span class="tooltip-action">TREND</span>
                <span class="tooltip-action">REFRESH</span>
                <span class="tooltip-action">DO_NOTHING</span>
                <span class="tooltip-action">FOLLOW</span>
                <span class="tooltip-action">MUTE</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Timeline Feed -->
      <div class="timeline-feed">
        <div class="timeline-axis"></div>
        
        <TransitionGroup name="timeline-item">
          <div 
            v-for="action in chronologicalActions" 
            :key="action._uniqueId || action.id || `${action.timestamp}-${action.agent_id}`" 
            class="timeline-item"
            :class="action.platform"
          >
            <div class="timeline-marker">
              <div class="marker-dot"></div>
            </div>
            
            <div class="timeline-card">
              <div class="card-header">
                <div class="agent-info">
                  <div class="avatar-placeholder">{{ (action.agent_name || 'A')[0] }}</div>
                  <span class="agent-name">{{ action.agent_name }}</span>
                </div>
                
                <div class="header-meta">
                  <div class="platform-indicator">
                    <svg v-if="action.platform === 'twitter'" viewBox="0 0 24 24" width="12" height="12" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"></path></svg>
                    <svg v-else viewBox="0 0 24 24" width="12" height="12" fill="currentColor"><path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.82 12.885c.014.109.014.219.014.328 0 3.36-3.51 6.087-7.834 6.087-4.324 0-7.834-2.727-7.834-6.087 0-.109 0-.219.014-.328a2.213 2.213 0 0 1-.71-1.643 2.221 2.221 0 0 1 2.222-2.222c.598 0 1.135.24 1.531.626 1.516-1.09 3.617-1.79 5.953-1.867l1.135-5.35a.161.161 0 0 1 .188-.128l3.762.792a1.484 1.484 0 1 1-.153.751l-3.417-.716-.996 4.699c2.302.096 4.363.79 5.859 1.867a2.16 2.16 0 0 1 1.531-.626 2.221 2.221 0 0 1 2.222 2.222c0 .724-.373 1.36-.938 1.735zM7.5 12.5c-.69 0-1.25.56-1.25 1.25s.56 1.25 1.25 1.25 1.25-.56 1.25-1.25-.56-1.25-1.25-1.25zm9.4 4.2c-.9.9-2.4 1.35-4.9 1.35-2.5 0-4-.45-4.9-1.35a.4.4 0 1 1 .566-.566c.71.71 1.9 1.116 4.334 1.116 2.434 0 3.624-.406 4.334-1.116a.4.4 0 1 1 .566.566zM16.5 15c-.69 0-1.25-.56-1.25-1.25s.56-1.25 1.25-1.25 1.25.56 1.25 1.25-.56 1.25-1.25 1.25z"></path></svg>
                  </div>
                  <div class="action-badge" :class="getActionTypeClass(action.action_type)">
                    {{ getActionTypeLabel(action.action_type) }}
                  </div>
                </div>
              </div>
              
              <div class="card-body">
                <!-- CREATE_POST: Publish post -->
                <div v-if="action.action_type === 'CREATE_POST' && action.action_args?.content" class="content-text main-text">
                  {{ action.action_args.content }}
                </div>

                <!-- QUOTE_POST: Quote post -->
                <template v-if="action.action_type === 'QUOTE_POST'">
                  <div v-if="action.action_args?.quote_content" class="content-text">
                    {{ action.action_args.quote_content }}
                  </div>
                  <div v-if="action.action_args?.original_content" class="quoted-block">
                    <div class="quote-header">
                      <svg class="icon-small" viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg>
                      <span class="quote-label">@{{ action.action_args.original_author_name || 'User' }}</span>
                    </div>
                    <div class="quote-text">
                      {{ truncateContent(action.action_args.original_content, 150) }}
                    </div>
                  </div>
                </template>

                <!-- REPOST: Repost post -->
                <template v-if="action.action_type === 'REPOST'">
                  <div class="repost-info">
                    <svg class="icon-small" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 1 21 5 17 9"></polyline><path d="M3 11V9a4 4 0 0 1 4-4h14"></path><polyline points="7 23 3 19 7 15"></polyline><path d="M21 13v2a4 4 0 0 1-4 4H3"></path></svg>
                    <span class="repost-label">Reposted from @{{ action.action_args?.original_author_name || 'User' }}</span>
                  </div>
                  <div v-if="action.action_args?.original_content" class="repost-content">
                    {{ truncateContent(action.action_args.original_content, 200) }}
                  </div>
                </template>

                <!-- LIKE_POST / DISLIKE_POST -->
                <template v-if="action.action_type === 'LIKE_POST' || action.action_type === 'DISLIKE_POST'">
                  <div class="like-info">
                    <svg v-if="action.action_type === 'LIKE_POST'" class="icon-small filled" viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg>
                    <svg v-else class="icon-small" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
                    <span class="like-label">{{ action.action_type === 'LIKE_POST' ? 'Liked' : 'Disliked' }} @{{ action.action_args?.post_author_name || 'User' }}'s post</span>
                  </div>
                  <div v-if="action.action_args?.post_content" class="liked-content">
                    "{{ truncateContent(action.action_args.post_content, 120) }}"
                  </div>
                </template>

                <!-- CREATE_COMMENT: Publish comment -->
                <template v-if="action.action_type === 'CREATE_COMMENT'">
                  <div v-if="action.action_args?.content" class="content-text">
                    {{ action.action_args.content }}
                  </div>
                  <div v-if="action.action_args?.post_content || action.action_args?.post_id" class="quoted-block">
                    <div class="quote-header">
                      <svg class="icon-small" viewBox="0 0 24 24" width="12" height="12" fill="currentColor"><path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.82 12.885c.014.109.014.219.014.328 0 3.36-3.51 6.087-7.834 6.087-4.324 0-7.834-2.727-7.834-6.087 0-.109 0-.219.014-.328a2.213 2.213 0 0 1-.71-1.643 2.221 2.221 0 0 1 2.222-2.222c.598 0 1.135.24 1.531.626 1.516-1.09 3.617-1.79 5.953-1.867l1.135-5.35a.161.161 0 0 1 .188-.128l3.762.792a1.484 1.484 0 1 1-.153.751l-3.417-.716-.996 4.699c2.302.096 4.363.79 5.859 1.867a2.16 2.16 0 0 1 1.531-.626 2.221 2.221 0 0 1 2.222 2.222c0 .724-.373 1.36-.938 1.735zM7.5 12.5c-.69 0-1.25.56-1.25 1.25s.56 1.25 1.25 1.25 1.25-.56 1.25-1.25-.56-1.25-1.25-1.25zm9.4 4.2c-.9.9-2.4 1.35-4.9 1.35-2.5 0-4-.45-4.9-1.35a.4.4 0 1 1 .566-.566c.71.71 1.9 1.116 4.334 1.116 2.434 0 3.624-.406 4.334-1.116a.4.4 0 1 1 .566.566zM16.5 15c-.69 0-1.25-.56-1.25-1.25s.56-1.25 1.25-1.25 1.25.56 1.25 1.25-.56 1.25-1.25 1.25z"></path></svg>
                      <span class="quote-label">Reply to @{{ action.action_args?.post_author_name || 'User' }}'s post{{ action.action_args?.post_id ? ` #${action.action_args.post_id}` : '' }}</span>
                    </div>
                    <div v-if="action.action_args?.post_content" class="quote-text">
                      {{ truncateContent(action.action_args.post_content, 150) }}
                    </div>
                  </div>
                </template>

                <!-- LIKE_COMMENT / DISLIKE_COMMENT -->
                <template v-if="action.action_type === 'LIKE_COMMENT' || action.action_type === 'DISLIKE_COMMENT'">
                  <div class="like-info">
                    <svg v-if="action.action_type === 'LIKE_COMMENT'" class="icon-small filled" viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg>
                    <svg v-else class="icon-small" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
                    <span class="like-label">{{ action.action_type === 'LIKE_COMMENT' ? 'Liked' : 'Disliked' }} @{{ action.action_args?.comment_author_name || 'User' }}'s comment</span>
                  </div>
                  <div v-if="action.action_args?.comment_content" class="liked-content">
                    "{{ truncateContent(action.action_args.comment_content, 120) }}"
                  </div>
                </template>

                <!-- SEARCH_POSTS: Search posts -->
                <template v-if="action.action_type === 'SEARCH_POSTS'">
                  <div class="search-info">
                    <svg class="icon-small" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                    <span class="search-label">Search Query:</span>
                    <span class="search-query">"{{ action.action_args?.query || '' }}"</span>
                  </div>
                </template>

                <!-- SEARCH_USER: Search users -->
                <template v-if="action.action_type === 'SEARCH_USER'">
                  <div class="search-info">
                    <svg class="icon-small" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                    <span class="search-label">Search User:</span>
                    <span class="search-query">"{{ action.action_args?.query || action.action_args?.user_id || '' }}"</span>
                  </div>
                </template>

                <!-- TREND: View trending posts -->
                <template v-if="action.action_type === 'TREND'">
                  <div class="search-info">
                    <svg class="icon-small" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>
                    <span class="search-label">Viewed trending posts</span>
                  </div>
                </template>

                <!-- REFRESH: Refresh feed -->
                <template v-if="action.action_type === 'REFRESH'">
                  <div class="idle-info">
                    <svg class="icon-small" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>
                    <span class="idle-label">Refreshed feed</span>
                  </div>
                </template>

                <!-- FOLLOW: Follow user -->
                <template v-if="action.action_type === 'FOLLOW'">
                  <div class="follow-info">
                    <svg class="icon-small" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="8.5" cy="7" r="4"></circle><line x1="20" y1="8" x2="20" y2="14"></line><line x1="23" y1="11" x2="17" y2="11"></line></svg>
                    <span class="follow-label">Followed @{{ action.action_args?.target_user_name || action.action_args?.user_id || 'User' }}</span>
                  </div>
                </template>

                <!-- MUTE: Mute user -->
                <template v-if="action.action_type === 'MUTE'">
                  <div class="follow-info">
                    <svg class="icon-small" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 5L6 9H2v6h4l5 4V5z"></path><line x1="23" y1="9" x2="17" y2="15"></line><line x1="17" y1="9" x2="23" y2="15"></line></svg>
                    <span class="follow-label">Muted @{{ action.action_args?.target_user_name || action.action_args?.user_id || 'User' }}</span>
                  </div>
                </template>

                <!-- DO_NOTHING: Không thao tác (im lặng) -->
                <template v-if="action.action_type === 'DO_NOTHING'">
                  <div class="idle-info">
                    <svg class="icon-small" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                    <span class="idle-label">Action Skipped</span>
                  </div>
                </template>

                <!-- Fallback chung: loại chưa biết hoặc có content nhưng chưa được xử lý ở trên -->
                <div v-if="!['CREATE_POST', 'QUOTE_POST', 'REPOST', 'LIKE_POST', 'DISLIKE_POST', 'CREATE_COMMENT', 'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER', 'TREND', 'REFRESH', 'FOLLOW', 'MUTE', 'DO_NOTHING'].includes(action.action_type) && action.action_args?.content" class="content-text">
                  {{ action.action_args.content }}
                </div>
              </div>

              <div class="card-footer">
                <span class="time-tag">R{{ action.round_num }} • {{ formatActionTime(action.timestamp) }}</span>
                <!-- Đã bỏ platform tag vì đã có ở header -->
              </div>
            </div>
          </div>
        </TransitionGroup>

        <div v-if="allActions.length === 0" class="waiting-state">
          <div class="pulse-ring"></div>
          <span>Waiting for agent actions...</span>
        </div>
      </div>
    </div>

    <!-- Bottom Info / Logs -->
    <div class="system-logs">
      <div class="log-header">
        <span class="log-title">SIMULATION MONITOR</span>
        <span class="log-id">{{ simulationId || 'NO_SIMULATION' }}</span>
      </div>
      <div class="log-content" ref="logContent">
        <div class="log-line" v-for="(log, idx) in systemLogs" :key="idx">
          <span class="log-time">{{ log.time }}</span>
          <span class="log-msg">{{ log.msg }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import {
  startSimulation,
  stopSimulation,
  getRunStatus,
  getRunStatusDetail,
  getSimulationActions
} from '../api/simulation'
import { generateReport } from '../api/report'

const props = defineProps({
  simulationId: String,
  maxRounds: Number, // max rounds được truyền từ Step2
  minutesPerRound: {
    type: Number,
    default: 30 // mặc định mỗi round là 30 phút
  },
  projectData: Object,
  graphData: Object,
  systemLogs: Array,
  previewOnly: Boolean    // preview: chỉ load kết quả mô phỏng cũ, không chạy lại
})

const emit = defineEmits(['go-back', 'next-step', 'add-log', 'update-status'])

const router = useRouter()

// State
const isGeneratingReport = ref(false)
const phase = ref(0) // 0: chưa bắt đầu, 1: đang chạy, 2: đã hoàn thành
const isStarting = ref(false)
const isStopping = ref(false)
const startError = ref(null)
const runStatus = ref({})
const allActions = ref([]) // toàn bộ action (cộng dồn tăng dần)
const actionIds = ref(new Set()) // tập ID action để khử trùng lặp
const scrollContainer = ref(null)
const actionTypeFilter = ref('ALL') // Bộ lọc timeline theo action_type

// Computed
// Danh sách action_type thực tế xuất hiện trong dữ liệu, dùng để dựng dropdown lọc
const availableActionTypes = computed(() => {
  const types = new Set(allActions.value.map(a => a.action_type).filter(Boolean))
  return Array.from(types).sort()
})

// Hiển thị action theo thứ tự thời gian (mới nhất ở cuối, tức phía dưới), có áp bộ lọc theo action_type
const chronologicalActions = computed(() => {
  if (actionTypeFilter.value === 'ALL') return allActions.value
  return allActions.value.filter(a => a.action_type === actionTypeFilter.value)
})

// Đếm action của từng platform
const twitterActionsCount = computed(() => {
  return allActions.value.filter(a => a.platform === 'twitter').length
})

const redditActionsCount = computed(() => {
  return allActions.value.filter(a => a.platform === 'reddit').length
})

// Format thời gian mô phỏng đã trôi qua (tính theo round và phút mỗi round)
const formatElapsedTime = (currentRound) => {
  if (!currentRound || currentRound <= 0) return '0h 0m'
  const totalMinutes = currentRound * props.minutesPerRound
  const hours = Math.floor(totalMinutes / 60)
  const minutes = totalMinutes % 60
  return `${hours}h ${minutes}m`
}

// Thời gian mô phỏng đã trôi qua của Twitter
const twitterElapsedTime = computed(() => {
  return formatElapsedTime(runStatus.value.twitter_current_round || 0)
})

// Thời gian mô phỏng đã trôi qua của Reddit
const redditElapsedTime = computed(() => {
  return formatElapsedTime(runStatus.value.reddit_current_round || 0)
})

// Methods
const addLog = (msg) => {
  emit('add-log', msg)
}

// Reset toàn bộ trạng thái (dùng để khởi động lại mô phỏng)
const resetAllState = () => {
  phase.value = 0
  runStatus.value = {}
  allActions.value = []
  actionIds.value = new Set()
  prevTwitterRound.value = 0
  prevRedditRound.value = 0
  startError.value = null
  isStarting.value = false
  isStopping.value = false
  stopPolling()  // dừng các polling cũ nếu có
}

// Khởi động mô phỏng
const doStartSimulation = async () => {
  if (!props.simulationId) {
    addLog('Error: missing simulationId')
    return
  }
  
  // Reset toàn bộ state trước để tránh bị ảnh hưởng từ lần mô phỏng trước
  resetAllState()
  
  isStarting.value = true
  startError.value = null
  addLog('Starting dual-platform parallel simulation...')
  emit('update-status', 'processing')
  
  try {
    const params = {
      simulation_id: props.simulationId,
      platform: 'parallel',
      force: true,  // ép bắt đầu lại từ đầu
      enable_graph_memory_update: true  // bật cập nhật graph động
    }
    
    if (props.maxRounds) {
      params.max_rounds = props.maxRounds
      addLog(`Set maximum simulation rounds: ${props.maxRounds}`)
    }
    
    addLog('Dynamic graph memory update mode enabled')
    
    const res = await startSimulation(params)
    
    if (res.success && res.data) {
      if (res.data.force_restarted) {
        addLog('✓ Old simulation logs cleared, restarting simulation')
      }
      addLog('✓ Simulation engine started successfully')
      addLog(`  ├─ PID: ${res.data.process_pid || '-'}`)
      
      phase.value = 1
      runStatus.value = res.data
      
      startStatusPolling()
      startDetailPolling()
    } else {
      startError.value = res.error || 'Start failed'
      addLog(`✗ Start failed: ${res.error || 'Unknown error'}`)
      emit('update-status', 'error')
    }
  } catch (err) {
    startError.value = err.message
    addLog(`✗ Start exception: ${err.message}`)
    emit('update-status', 'error')
  } finally {
    isStarting.value = false
  }
}

// Dừng mô phỏng
const handleStopSimulation = async () => {
  if (!props.simulationId) return
  
  isStopping.value = true
  addLog('Stopping simulation...')
  
  try {
    const res = await stopSimulation({ simulation_id: props.simulationId })
    
    if (res.success) {
      addLog('✓ Simulation stopped')
      phase.value = 2
      stopPolling()
      emit('update-status', 'completed')
    } else {
      addLog(`Stop failed: ${res.error || 'Unknown error'}`)
    }
  } catch (err) {
    addLog(`Stop exception: ${err.message}`)
  } finally {
    isStopping.value = false
  }
}

// Polling trạng thái
let statusTimer = null
let detailTimer = null

const startStatusPolling = () => {
  statusTimer = setInterval(fetchRunStatus, 2000)
}

const startDetailPolling = () => {
  detailTimer = setInterval(fetchRunStatusDetail, 3000)
}

const stopPolling = () => {
  if (statusTimer) {
    clearInterval(statusTimer)
    statusTimer = null
  }
  if (detailTimer) {
    clearInterval(detailTimer)
    detailTimer = null
  }
}

// Theo dõi round trước đó của từng platform để phát hiện thay đổi và ghi log
const prevTwitterRound = ref(0)
const prevRedditRound = ref(0)

const fetchRunStatus = async () => {
  if (!props.simulationId) return
  
  try {
    const res = await getRunStatus(props.simulationId)
    
    if (res.success && res.data) {
      const data = res.data
      
      runStatus.value = data
      
      // Kiểm tra thay đổi round của từng platform và ghi log
      if (data.twitter_current_round > prevTwitterRound.value) {
        addLog(`[Plaza] R${data.twitter_current_round}/${data.total_rounds} | T:${data.twitter_simulated_hours || 0}h | A:${data.twitter_actions_count}`)
        prevTwitterRound.value = data.twitter_current_round
      }
      
      if (data.reddit_current_round > prevRedditRound.value) {
        addLog(`[Community] R${data.reddit_current_round}/${data.total_rounds} | T:${data.reddit_simulated_hours || 0}h | A:${data.reddit_actions_count}`)
        prevRedditRound.value = data.reddit_current_round
      }
      
      // Kiểm tra mô phỏng đã hoàn thành chưa (theo runner_status hoặc trạng thái hoàn thành của platform)
      const isCompleted = data.runner_status === 'completed' || data.runner_status === 'stopped'
      
      // Kiểm tra thêm: nếu backend chưa kịp cập nhật runner_status nhưng platform đã báo xong
      // Dựa vào twitter_completed và reddit_completed để xác định
      const platformsCompleted = checkPlatformsCompleted(data)
      
      if (isCompleted || platformsCompleted) {
        if (platformsCompleted && !isCompleted) {
          addLog('✓ Detected all platform simulations have finished')
        }
        addLog('✓ Simulation completed')
        phase.value = 2
        stopPolling()
        emit('update-status', 'completed')
      }
    }
  } catch (err) {
    console.warn('Failed to fetch run status:', err)
  }
}

// Kiểm tra tất cả platform đang bật đã hoàn thành chưa
const checkPlatformsCompleted = (data) => {
  // Nếu không có dữ liệu platform nào thì trả về false
  if (!data) return false
  
  // Kiểm tra trạng thái hoàn thành của từng platform
  const twitterCompleted = data.twitter_completed === true
  const redditCompleted = data.reddit_completed === true
  
  // Nếu có ít nhất một platform hoàn thành, kiểm tra xem tất cả platform đang bật đã hoàn thành chưa
  // Dùng actions_count để suy ra platform có được bật không (count > 0 hoặc từng running)
  const twitterEnabled = (data.twitter_actions_count > 0) || data.twitter_running || twitterCompleted
  const redditEnabled = (data.reddit_actions_count > 0) || data.reddit_running || redditCompleted
  
  // Nếu không có platform nào được bật thì trả về false
  if (!twitterEnabled && !redditEnabled) return false
  
  // Kiểm tra tất cả platform đang bật đã hoàn thành chưa
  if (twitterEnabled && !twitterCompleted) return false
  if (redditEnabled && !redditCompleted) return false
  
  return true
}

const fetchRunStatusDetail = async () => {
  if (!props.simulationId) return
  
  try {
    const res = await getRunStatusDetail(props.simulationId)
    
    if (res.success && res.data) {
      // Dùng all_actions để lấy đầy đủ danh sách action
      const serverActions = res.data.all_actions || []
      
      // Thêm tăng dần các action mới (khử trùng lặp)
      let newActionsAdded = 0
      serverActions.forEach(action => {
        // Tạo unique ID
        const actionId = action.id || `${action.timestamp}-${action.platform}-${action.agent_id}-${action.action_type}`
        
        if (!actionIds.value.has(actionId)) {
          actionIds.value.add(actionId)
          allActions.value.push({
            ...action,
            _uniqueId: actionId
          })
          newActionsAdded++
        }
      })
      
      // Không tự động cuộn, để người dùng tự do xem timeline
      // Action mới sẽ được thêm ở phía dưới
    }
  } catch (err) {
    console.warn('Failed to fetch detailed status:', err)
  }
}

// Preview: load action cũ theo từng trang (phân trang) để tránh kéo toàn bộ 1 lần.
// Backend /actions trả { count, actions }; còn dữ liệu khi count === PAGE_SIZE.
const PAGE_SIZE = 200
const isLoadingActions = ref(false)

const appendActions = (serverActions) => {
  serverActions.forEach(action => {
    const actionId = action.id || `${action.timestamp}-${action.platform}-${action.agent_id}-${action.action_type}`
    if (!actionIds.value.has(actionId)) {
      actionIds.value.add(actionId)
      allActions.value.push({ ...action, _uniqueId: actionId })
    }
  })
}

const loadActionsPaged = async () => {
  if (!props.simulationId) return
  isLoadingActions.value = true
  let offset = 0
  try {
    // get_actions sort timestamp giảm dần → nạp ngược để allActions cuối cùng theo thứ tự thời gian tăng dần
    const pages = []
    while (true) {
      const res = await getSimulationActions(props.simulationId, { limit: PAGE_SIZE, offset })
      if (!res.success || !res.data) break
      const batch = res.data.actions || []
      pages.push(batch)
      addLog(`Loaded ${offset + batch.length} actions...`)
      if (batch.length < PAGE_SIZE) break
      offset += PAGE_SIZE
    }
    // pages[0] mới nhất → đảo để cũ nhất vào trước
    for (let i = pages.length - 1; i >= 0; i--) {
      appendActions([...pages[i]].reverse())
    }
    addLog(`✓ Loaded ${allActions.value.length} actions total`)
    phase.value = 2
    emit('update-status', 'completed')
  } catch (err) {
    addLog(`✗ Failed to load actions: ${err.message}`)
    emit('update-status', 'error')
  } finally {
    isLoadingActions.value = false
  }
}

// Helpers
const getActionTypeLabel = (type) => {
  return type || 'UNKNOWN'
}

const getActionTypeClass = (type) => {
  const classes = {
    'CREATE_POST': 'badge-post',
    'REPOST': 'badge-action',
    'LIKE_POST': 'badge-action',
    'DISLIKE_POST': 'badge-action',
    'CREATE_COMMENT': 'badge-comment',
    'LIKE_COMMENT': 'badge-action',
    'DISLIKE_COMMENT': 'badge-action',
    'QUOTE_POST': 'badge-post',
    'FOLLOW': 'badge-meta',
    'MUTE': 'badge-meta',
    'SEARCH_POSTS': 'badge-meta',
    'SEARCH_USER': 'badge-meta',
    'TREND': 'badge-meta',
    'REFRESH': 'badge-meta',
    'DO_NOTHING': 'badge-idle'
  }
  return classes[type] || 'badge-default'
}

const truncateContent = (content, maxLength = 100) => {
  if (!content) return ''
  if (content.length > maxLength) return content.substring(0, maxLength) + '...'
  return content
}

const formatActionTime = (timestamp) => {
  if (!timestamp) return ''
  try {
    return new Date(timestamp).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
  } catch {
    return ''
  }
}

const handleNextStep = async () => {
  if (!props.simulationId) {
    addLog('Error: missing simulationId')
    return
  }
  
  if (isGeneratingReport.value) {
    addLog('Report generation request has been sent, please wait...')
    return
  }
  
  isGeneratingReport.value = true
  addLog('Starting report generation...')
  
  try {
    const res = await generateReport({
      simulation_id: props.simulationId,
      force_regenerate: true
    })
    
    if (res.success && res.data) {
      const reportId = res.data.report_id
      addLog(`✓ Report generation task started: ${reportId}`)
      
      // Chuyển sang trang report
      router.push({ name: 'Report', params: { reportId } })
    } else {
      addLog(`✗ Failed to start report generation: ${res.error || 'Unknown error'}`)
      isGeneratingReport.value = false
    }
  } catch (err) {
    addLog(`✗ Report generation exception: ${err.message}`)
    isGeneratingReport.value = false
  }
}

// Scroll log xuống cuối
const logContent = ref(null)
watch(() => props.systemLogs?.length, () => {
  nextTick(() => {
    if (logContent.value) {
      logContent.value.scrollTop = logContent.value.scrollHeight
    }
  })
})

onMounted(() => {
  if (!props.simulationId) return

  // Preview/read-only mode: chỉ đọc kết quả mô phỏng đã chạy trước đó,
  // KHÔNG gọi startSimulation (không chạy lại mô phỏng thật).
  // Dùng run-status (nhẹ) + actions phân trang thay vì run-status/detail
  // (endpoint detail đọc toàn bộ file 4 lần & nhân 3 payload → rất chậm).
  if (props.previewOnly) {
    addLog('Step3 (preview): loading existing results (paged), no re-run')
    fetchRunStatus()
    loadActionsPaged()
    return
  }

  addLog('Step3 simulation run initialization')
  doStartSimulation()
})

onUnmounted(() => {
  stopPolling()
})
</script>

<style scoped>
.simulation-panel {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: #FFFFFF;
  font-family: 'Space Grotesk', 'Noto Sans SC', system-ui, sans-serif;
  overflow: hidden;
}

/* --- Control Bar --- */
.control-bar {
  background: #FFF;
  padding: 12px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #EAEAEA;
  z-index: 10;
  min-height: 64px;
}

.status-group {
  display: flex;
  gap: 60px;
  justify-content: center;
}

/* Platform Status Cards */
.platform-status {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 6px 12px;
  border-radius: 4px;
  background: #FAFAFA;
  border: 1px solid #EAEAEA;
  opacity: 0.7;
  transition: all 0.3s;
  min-width: 180px;
  position: relative;
  cursor: pointer;
}

.platform-status.active {
  opacity: 1;
  border-color: #333;
  background: #FFF;
}

.platform-status.completed {
  opacity: 1;
  border-color: #1A936F;
  background: #F2FAF6;
}

/* Actions Tooltip */
.actions-tooltip {
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  margin-top: 8px;
  padding: 10px 14px;
  background: #000;
  color: #FFF;
  border-radius: 4px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  opacity: 0;
  visibility: hidden;
  transition: all 0.2s ease;
  z-index: 100;
  min-width: 180px;
  pointer-events: none;
}

.actions-tooltip::before {
  content: '';
  position: absolute;
  top: -6px;
  left: 50%;
  transform: translateX(-50%);
  border-left: 6px solid transparent;
  border-right: 6px solid transparent;
  border-bottom: 6px solid #000;
}

.platform-status:hover .actions-tooltip {
  opacity: 1;
  visibility: visible;
}

.tooltip-title {
  font-size: 10px;
  font-weight: 600;
  color: #999;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
}

.tooltip-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.tooltip-action {
  font-size: 10px;
  font-weight: 600;
  padding: 3px 8px;
  background: rgba(255, 255, 255, 0.15);
  border-radius: 2px;
  color: #FFF;
  letter-spacing: 0.03em;
}

.platform-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 2px;
}

.platform-name {
  font-size: 11px;
  font-weight: 700;
  color: #000;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.platform-status.twitter .platform-icon { color: #000; }
.platform-status.reddit .platform-icon { color: #000; }

.platform-stats {
  display: flex;
  gap: 10px;
}

.stat {
  display: flex;
  align-items: baseline;
  gap: 3px;
}

.stat-label {
  font-size: 8px;
  color: #999;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.stat-value {
  font-size: 11px;
  font-weight: 600;
  color: #333;
}

.stat-total, .stat-unit {
  font-size: 9px;
  color: #999;
  font-weight: 400;
}

.status-badge {
  margin-left: auto;
  color: #1A936F;
  display: flex;
  align-items: center;
}

/* Action Button */
.action-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  font-size: 13px;
  font-weight: 600;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s ease;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.action-btn.primary {
  background: #000;
  color: #FFF;
}

.action-btn.primary:hover:not(:disabled) {
  background: #333;
}

.action-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

/* --- Main Content Area --- */
.main-content-area {
  flex: 1;
  overflow-y: auto;
  position: relative;
  background: #FFF;
}

/* Timeline Header */
.timeline-header {
  position: sticky;
  top: 0;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(8px);
  padding: 16px 24px;
  border-bottom: 1px solid #EAEAEA;
  z-index: 5;
  display: flex;
  justify-content: center;
}

.timeline-stats {
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 11px;
  color: #666;
  background: #F5F5F5;
  padding: 4px 12px;
  border-radius: 20px;
}

.total-count {
  font-weight: 600;
  color: #333;
}

.action-type-filter {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 600;
  color: #333;
  background: #FFF;
  border: 1px solid #DDD;
  border-radius: 12px;
  padding: 3px 8px;
  cursor: pointer;
}

.action-type-filter:hover { border-color: #999; }

.platform-breakdown {
  display: flex;
  align-items: center;
  gap: 8px;
}

.breakdown-item {
  display: flex;
  align-items: center;
  gap: 4px;
}

.breakdown-divider { color: #DDD; }
.breakdown-item.twitter { color: #000; }
.breakdown-item.reddit { color: #000; }

/* --- Timeline Feed --- */
.timeline-feed {
  padding: 24px 0;
  position: relative;
  min-height: 100%;
  max-width: 900px;
  margin: 0 auto;
}

.timeline-axis {
  position: absolute;
  left: 50%;
  top: 0;
  bottom: 0;
  width: 1px;
  background: #EAEAEA; /* đường line gọn hơn */
  transform: translateX(-50%);
}

.timeline-item {
  display: flex;
  justify-content: center;
  margin-bottom: 32px;
  position: relative;
  width: 100%;
}

.timeline-marker {
  position: absolute;
  left: 50%;
  top: 24px;
  width: 10px;
  height: 10px;
  background: #FFF;
  border: 1px solid #CCC;
  border-radius: 50%;
  transform: translateX(-50%);
  z-index: 2;
  display: flex;
  align-items: center;
  justify-content: center;
}

.marker-dot {
  width: 4px;
  height: 4px;
  background: #CCC;
  border-radius: 50%;
}

.timeline-item.twitter .marker-dot { background: #000; }
.timeline-item.reddit .marker-dot { background: #000; }
.timeline-item.twitter .timeline-marker { border-color: #000; }
.timeline-item.reddit .timeline-marker { border-color: #000; }

/* Card Layout */
.timeline-card {
  width: calc(100% - 48px);
  background: #FFF;
  border-radius: 2px;
  padding: 16px 20px;
  border: 1px solid #EAEAEA;
  box-shadow: 0 2px 10px rgba(0,0,0,0.02);
  position: relative;
  transition: all 0.2s;
}

.timeline-card:hover {
  box-shadow: 0 4px 12px rgba(0,0,0,0.05);
  border-color: #DDD;
}

/* Left side (Twitter) */
.timeline-item.twitter {
  justify-content: flex-start;
  padding-right: 50%;
}
.timeline-item.twitter .timeline-card {
  margin-left: auto;
  margin-right: 32px; /* khoảng cách tới trục */
}

/* Right side (Reddit) */
.timeline-item.reddit {
  justify-content: flex-end;
  padding-left: 50%;
}
.timeline-item.reddit .timeline-card {
  margin-right: auto;
  margin-left: 32px; /* khoảng cách tới trục */
}

/* Card Content Styles */
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid #F5F5F5;
}

.agent-info {
  display: flex;
  align-items: center;
  gap: 10px;
}

.avatar-placeholder {
  width: 24px;
  height: 24px;
  background: #000;
  color: #FFF;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}

.agent-name {
  font-size: 13px;
  font-weight: 600;
  color: #000;
}

.header-meta {
  display: flex;
  align-items: center;
  gap: 8px;
}

.platform-indicator {
  color: #999;
  display: flex;
  align-items: center;
}

.action-badge {
  font-size: 9px;
  padding: 2px 6px;
  border-radius: 2px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border: 1px solid transparent;
}

/* Monochromatic Badges */
.badge-post { background: #F0F0F0; color: #333; border-color: #E0E0E0; }
.badge-comment { background: #F0F0F0; color: #666; border-color: #E0E0E0; }
.badge-action { background: #FFF; color: #666; border: 1px solid #E0E0E0; }
.badge-meta { background: #FAFAFA; color: #999; border: 1px dashed #DDD; }
.badge-idle { opacity: 0.5; }

.content-text {
  font-size: 13px;
  line-height: 1.6;
  color: #333;
  margin-bottom: 10px;
}

.content-text.main-text {
  font-size: 14px;
  color: #000;
}

/* Info Blocks (Quote, Repost, etc) */
.quoted-block, .repost-content {
  background: #F9F9F9;
  border: 1px solid #EEE;
  padding: 10px 12px;
  border-radius: 2px;
  margin-top: 8px;
  font-size: 12px;
  color: #555;
}

.quote-header, .repost-info, .like-info, .search-info, .follow-info, .vote-info, .idle-info, .comment-context {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
  font-size: 11px;
  color: #666;
}

.icon-small {
  color: #999;
}
.icon-small.filled {
  color: #999; /* giữ icon trung tính nếu không cần highlight */
}

.search-query {
  font-family: 'JetBrains Mono', monospace;
  background: #F0F0F0;
  padding: 0 4px;
  border-radius: 2px;
}

.card-footer {
  margin-top: 12px;
  display: flex;
  justify-content: flex-end;
  font-size: 10px;
  color: #BBB;
  font-family: 'JetBrains Mono', monospace;
}

/* Waiting State */
.waiting-state {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  color: #CCC;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.pulse-ring {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: 1px solid #EAEAEA;
  animation: ripple 2s infinite;
}

@keyframes ripple {
  0% { transform: scale(0.8); opacity: 1; border-color: #CCC; }
  100% { transform: scale(2.5); opacity: 0; border-color: #EAEAEA; }
}

/* Animation */
.timeline-item-enter-active,
.timeline-item-leave-active {
  transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
}

.timeline-item-enter-from {
  opacity: 0;
  transform: translateY(20px);
}

.timeline-item-leave-to {
  opacity: 0;
}

/* Logs */
.system-logs {
  background: #000;
  color: #DDD;
  padding: 16px;
  font-family: 'JetBrains Mono', monospace;
  border-top: 1px solid #222;
  flex-shrink: 0;
}

.log-header {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid #333;
  padding-bottom: 8px;
  margin-bottom: 8px;
  font-size: 10px;
  color: #666;
}

.log-content {
  display: flex;
  flex-direction: column;
  gap: 4px;
  height: 100px;
  overflow-y: auto;
  padding-right: 4px;
}

.log-content::-webkit-scrollbar { width: 4px; }
.log-content::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }

.log-line {
  font-size: 11px;
  display: flex;
  gap: 12px;
  line-height: 1.5;
}

.log-time { color: #555; min-width: 75px; }
.log-msg { color: #BBB; word-break: break-all; }
.mono { font-family: 'JetBrains Mono', monospace; }

/* Loading spinner for button */
.loading-spinner-small {
  display: inline-block;
  width: 14px;
  height: 14px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: #FFF;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-right: 6px;
}
</style>