import { useState } from 'react'
import {
  ArrowRight,
  Braces,
  Check,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  Database,
  FileCode2,
  Folder,
  FolderOpen,
  LoaderCircle,
  RefreshCw,
  Search,
  Sparkles,
  TerminalSquare,
  X,
} from 'lucide-react'
import './App.css'

type TreeNode = {
  name: string
  type: 'directory' | 'file'
  path: string
  children?: TreeNode[]
}

type Manifest = {
  repo_id: string
  indexed_at: string
  file_count: number
  chunk_count: number
  embedding_model: string
}

type Source = {
  file_path: string
  language: string
  start_line: number
  end_line: number
  symbol_name?: string | null
  score: number
  content: string
}

type RepoStatus = {
  repo_id: string
  repo_path: string
  tree: TreeNode
  index_available: boolean
  manifest?: Manifest | null
}

type QueryResponse = {
  answer: string
  abstained: boolean
  answer_mode: 'llm' | 'retrieval' | 'abstained'
  answer_notice?: string | null
  sources: Source[]
}

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

async function postJson<T>(path: string, body: object): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const payload = await response.json()
  if (!response.ok) {
    throw new Error(payload.detail ?? 'Request failed')
  }
  return payload
}

function nodeContainsMatch(node: TreeNode, matchedFiles: Map<string, number>): boolean {
  if (node.type === 'file') {
    return matchedFiles.has(node.path)
  }
  return node.children?.some((child) => nodeContainsMatch(child, matchedFiles)) ?? false
}

function TreeItem({
  node,
  depth,
  matchedFiles,
  selectedFile,
  onFileClick,
}: {
  node: TreeNode
  depth: number
  matchedFiles: Map<string, number>
  selectedFile: string | null
  onFileClick: (path: string) => void
}) {
  const containsMatch = nodeContainsMatch(node, matchedFiles)
  const [open, setOpen] = useState(depth < 2 || containsMatch)
  const expanded = open || containsMatch
  const rank = matchedFiles.get(node.path)

  if (node.type === 'directory') {
    return (
      <div>
        <button
          className="tree-row directory-row"
          style={{ paddingLeft: `${10 + depth * 14}px` }}
          onClick={() => setOpen((value) => !value)}
          type="button"
        >
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          {expanded ? <FolderOpen size={15} /> : <Folder size={15} />}
          <span>{node.name}</span>
        </button>
        {expanded &&
          node.children?.map((child) => (
            <TreeItem
              key={child.path}
              node={child}
              depth={depth + 1}
              matchedFiles={matchedFiles}
              selectedFile={selectedFile}
              onFileClick={onFileClick}
            />
          ))}
      </div>
    )
  }

  return (
    <button
      className={`tree-row file-row ${selectedFile === node.path ? 'selected' : ''} ${rank ? 'matched' : ''}`}
      style={{ paddingLeft: `${28 + depth * 14}px` }}
      onClick={() => onFileClick(node.path)}
      type="button"
    >
      <FileCode2 size={14} />
      <span>{node.name}</span>
      {rank && <b className="rank-chip">#{rank}</b>}
    </button>
  )
}

function CodeViewer({
  filePath,
  content,
  source,
  onClose,
}: {
  filePath: string
  content: string
  source?: Source
  onClose: () => void
}) {
  const lines = content.split('\n')
  return (
    <section className="code-viewer">
      <header>
        <div>
          <span className="eyebrow">READ-ONLY SOURCE</span>
          <h2>{filePath}</h2>
        </div>
        <button type="button" className="icon-button" onClick={onClose} aria-label="Close file viewer">
          <X size={18} />
        </button>
      </header>
      <div className="code-scroll">
        {lines.map((line, index) => {
          const lineNumber = index + 1
          const highlighted = source && lineNumber >= source.start_line && lineNumber <= source.end_line
          return (
            <div className={`code-line ${highlighted ? 'highlighted' : ''}`} key={`${lineNumber}-${line}`}>
              <span className="line-number">{lineNumber}</span>
              <code>{line || ' '}</code>
            </div>
          )
        })}
      </div>
    </section>
  )
}

function App() {
  const [repoPath, setRepoPath] = useState('./sample_project')
  const [status, setStatus] = useState<RepoStatus | null>(null)
  const [question, setQuestion] = useState('Where is the login feature?')
  const [useLlm, setUseLlm] = useState(true)
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [loading, setLoading] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [fileContent, setFileContent] = useState('')
  const [selectedSource, setSelectedSource] = useState<Source | undefined>()
  const [indexActive, setIndexActive] = useState(false)

  const matchedFiles = new Map(result?.sources.map((source, index) => [source.file_path, index + 1]) ?? [])

  async function scanRepository() {
    setLoading('scan')
    setError(null)
    setResult(null)
    setIndexActive(false)
    try {
      const nextStatus = await postJson<RepoStatus>('/api/repo/status', { repo_path: repoPath })
      setStatus(nextStatus)
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Could not scan repository')
    } finally {
      setLoading(null)
    }
  }

  async function rebuildIndex() {
    setLoading('index')
    setError(null)
    try {
      await postJson('/api/repo/index', { repo_path: repoPath })
      const nextStatus = await postJson<RepoStatus>('/api/repo/status', { repo_path: repoPath })
      setStatus(nextStatus)
      setIndexActive(true)
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Could not build index')
    } finally {
      setLoading(null)
    }
  }

  async function askQuestion(llmMode = useLlm) {
    if (!indexActive) return
    setLoading('query')
    setError(null)
    setSelectedFile(null)
    try {
      const nextResult = await postJson<QueryResponse>('/api/query', {
        repo_path: repoPath,
        question,
        use_llm: llmMode,
      })
      setResult(nextResult)
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Query failed')
    } finally {
      setLoading(null)
    }
  }

  async function openFile(filePath: string, source?: Source) {
    setLoading('file')
    setError(null)
    try {
      const payload = await postJson<{ content: string }>('/api/file', {
        repo_path: repoPath,
        file_path: filePath,
      })
      setSelectedFile(filePath)
      setSelectedSource(source ?? result?.sources.find((item) => item.file_path === filePath))
      setFileContent(payload.content)
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Could not open file')
    } finally {
      setLoading(null)
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Braces size={22} /></div>
          <div>
            <strong>Evidence Explorer</strong>
            <span>Codebase RAG workbench</span>
          </div>
        </div>
        <div className="system-state">
          <span className="pulse" />
          Hybrid retrieval online
          <kbd>⌘ K</kbd>
        </div>
      </header>

      <div className="workspace">
        <aside className="repo-panel">
          <div className="panel-heading">
            <span className="eyebrow">01 / REPOSITORY</span>
            <h1>Ground the search.</h1>
            <p>Inspect a local codebase, reuse its index, or rebuild after changes.</p>
          </div>

          <label className="path-label" htmlFor="repo-path">Local repository path</label>
          <div className="path-control">
            <TerminalSquare size={17} />
            <input id="repo-path" value={repoPath} onChange={(event) => setRepoPath(event.target.value)} />
            <button type="button" onClick={scanRepository} disabled={loading !== null} aria-label="Scan repository">
              {loading === 'scan' ? <LoaderCircle className="spin" size={17} /> : <ArrowRight size={17} />}
            </button>
          </div>

          <div className={`index-card ${status?.index_available ? 'ready' : ''}`}>
            <div className="index-icon">
              {status?.index_available ? <Check size={18} /> : <Database size={18} />}
            </div>
            <div>
              <strong>{status?.index_available ? 'Index ready' : 'No active index'}</strong>
              <span>
                {status?.manifest
                  ? `${status.manifest.file_count} files · ${status.manifest.chunk_count} chunks`
                  : 'Scan a path to check availability'}
              </span>
            </div>
            {status?.manifest && (
              <time>{new Date(status.manifest.indexed_at).toLocaleDateString()}</time>
            )}
          </div>

          <div className="repo-actions">
            <button
              className="secondary-button"
              type="button"
              disabled={!status?.index_available}
              onClick={() => setIndexActive(true)}
            >
              <Database size={15} /> Use existing
            </button>
            <button className="primary-button" type="button" onClick={rebuildIndex} disabled={!status || loading !== null}>
              {loading === 'index' ? <LoaderCircle className="spin" size={15} /> : <RefreshCw size={15} />}
              Rebuild index
            </button>
          </div>

          <div className="tree-header">
            <span>PROJECT TREE</span>
            {status?.tree && <small>{status.repo_id}</small>}
          </div>
          <div className="tree-panel">
            {status?.tree ? (
              <TreeItem
                node={status.tree}
                depth={0}
                matchedFiles={matchedFiles}
                selectedFile={selectedFile}
                onFileClick={(path) => openFile(path)}
              />
            ) : (
              <div className="empty-tree">
                <Folder size={28} />
                <p>The filtered project structure will appear here.</p>
              </div>
            )}
          </div>
        </aside>

        <section className="main-panel">
          {selectedFile ? (
            <CodeViewer
              filePath={selectedFile}
              content={fileContent}
              source={selectedSource}
              onClose={() => setSelectedFile(null)}
            />
          ) : (
            <>
              <section className="query-hero">
                <div className="query-heading">
                  <div>
                    <span className="eyebrow">02 / ASK THE CODEBASE</span>
                    <h2>Find the implementation, not just the words.</h2>
                  </div>
                  <label className="mode-switch">
                    <input
                      type="checkbox"
                      checked={useLlm}
                      disabled={loading !== null}
                      onChange={(event) => setUseLlm(event.target.checked)}
                    />
                    <span />
                    LLM answer
                  </label>
                </div>
                <div className="query-control">
                  <Search size={20} />
                  <textarea
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder="Ask where or how a feature is implemented..."
                    rows={2}
                  />
                  <button
                    type="button"
                    onClick={() => askQuestion()}
                    disabled={!indexActive || !question.trim() || loading !== null}
                  >
                    {loading === 'query' ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}
                    Ask
                  </button>
                </div>
                <div className="suggestions">
                  {['Where is login handled?', 'Where is markdown rendered?', 'How is branch cache implemented?'].map((item) => (
                    <button type="button" key={item} onClick={() => setQuestion(item)}>{item}</button>
                  ))}
                </div>
              </section>

              {error && (
                <div className="error-banner">
                  <CircleAlert size={18} />
                  <span>{error}</span>
                </div>
              )}

              {result ? (
                <section className="results">
                  <article className={`answer-card ${result.abstained ? 'abstained' : ''}`}>
                    <header>
                      <span className="eyebrow">{result.abstained ? 'INSUFFICIENT EVIDENCE' : 'GROUNDED ANSWER'}</span>
                      <strong>
                        {result.answer_mode === 'llm'
                          ? 'LLM answer grounded in retrieved source.'
                          : result.answer_mode === 'retrieval'
                            ? 'Retrieval-only answer.'
                            : 'The index cannot support this answer.'}
                      </strong>
                    </header>
                    {result.answer_notice && <p className="answer-notice">{result.answer_notice}</p>}
                    <pre>{result.answer}</pre>
                  </article>

                  {!result.abstained && (
                    <>
                      <div className="source-heading">
                        <div>
                          <span className="eyebrow">03 / RANKED EVIDENCE</span>
                          <h2>{result.sources.length} source chunks retrieved</h2>
                        </div>
                        <span className="fusion-badge">Dense + BM25 + rerank</span>
                      </div>
                      <div className="source-grid">
                        {result.sources.map((source, index) => (
                          <button
                            type="button"
                            className="source-card"
                            key={`${source.file_path}-${source.start_line}`}
                            onClick={() => openFile(source.file_path, source)}
                          >
                            <div className="source-rank">{String(index + 1).padStart(2, '0')}</div>
                            <div className="source-body">
                              <strong>{source.file_path}</strong>
                              <span>
                                {source.symbol_name ?? source.language} · lines {source.start_line}-{source.end_line}
                              </span>
                              <code>{source.content.split('\n').slice(0, 3).join('\n')}</code>
                            </div>
                            <div className="score">
                              <b>{Math.round(source.score * 100)}</b>
                              <span>relative rank</span>
                            </div>
                          </button>
                        ))}
                      </div>
                    </>
                  )}
                </section>
              ) : (
                <section className="empty-state">
                  <div className="orbit orbit-one" />
                  <div className="orbit orbit-two" />
                  <div className="empty-icon"><Search size={28} /></div>
                  <h2>Evidence will collect here.</h2>
                  <p>Scan a repository, use or rebuild its index, then ask a code-navigation question.</p>
                  <div className="pipeline">
                    <span>Dense</span><i /><span>BM25</span><i /><span>Rerank</span><i /><span>Ground</span>
                  </div>
                </section>
              )}
            </>
          )}
        </section>
      </div>
    </main>
  )
}

export default App
