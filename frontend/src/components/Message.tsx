/**
 * Message Component with Markdown and Syntax Highlighting
 * 
 * @author Adrian Johnson <adrian207@gmail.com>
 */
import { User, Bot, AlertCircle } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { format } from 'date-fns'
import type { Message as MessageType } from '@/types'
import clsx from 'clsx'

interface MessageProps {
  message: MessageType
}

export default function Message({ message }: MessageProps) {
  const isUser = message.role === 'user'

  return (
    <div
      className={clsx(
        'flex gap-4',
        isUser ? 'flex-row-reverse' : 'flex-row'
      )}
    >
      <div
        className={clsx(
          'flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center',
          isUser
            ? 'bg-primary-600 text-white'
            : message.error
            ? 'bg-red-600 text-white'
            : 'bg-gray-700 text-white'
        )}
      >
        {isUser ? (
          <User className="w-5 h-5" />
        ) : message.error ? (
          <AlertCircle className="w-5 h-5" />
        ) : (
          <Bot className="w-5 h-5" />
        )}
      </div>

      <div className={clsx('flex-1 max-w-3xl', isUser ? 'text-right' : 'text-left')}>
        <div className="flex items-center gap-2 mb-1">
          {!isUser && message.model && (
            <span className="text-xs text-gray-500 dark:text-gray-400 font-mono">
              {message.model}
            </span>
          )}
          {message.cached && (
            <span className="text-xs bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 px-2 py-0.5 rounded-full">
              Cached
            </span>
          )}
          <span className="text-xs text-gray-400">
            {format(message.timestamp, 'HH:mm:ss')}
          </span>
        </div>

        <div
          className={clsx(
            'rounded-lg p-4',
            isUser
              ? 'bg-primary-600 text-white inline-block'
              : message.error
              ? 'bg-red-50 dark:bg-red-900/20 text-red-900 dark:text-red-200'
              : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700'
          )}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <ReactMarkdown
              className="markdown-body prose dark:prose-invert max-w-none"
              components={{
                code({ inline, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '')
                  return !inline && match ? (
                    <SyntaxHighlighter
                      style={vscDarkPlus}
                      language={match[1]}
                      PreTag="div"
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  ) : (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  )
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>
      </div>
    </div>
  )
}

