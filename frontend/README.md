# Dual RAG LLM Frontend

**Modern React + TypeScript frontend for the Dual RAG LLM System**

**Author:** Adrian Johnson <adrian207@gmail.com>  
**Version:** 1.13.0

## Features

✅ **Modern Stack**
- React 18 with TypeScript
- Vite for lightning-fast development
- Tailwind CSS for styling
- Zustand for state management
- React Query for data fetching

✅ **Real-time Streaming**
- Token-by-token response streaming
- EventSource API integration
- Smooth animations

✅ **Rich UI Components**
- Markdown rendering with react-markdown
- Syntax highlighting for 22+ languages
- Dark/Light/System theme support
- Responsive design

✅ **Advanced Features**
- Model selection
- Web search toggle
- GitHub integration
- System statistics dashboard
- Cache management

## Quick Start

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

Frontend will be available at http://localhost:3001

### Build for Production

```bash
npm run build
```

Output will be in `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/        # React components
│   │   ├── Header.tsx
│   │   ├── Sidebar.tsx
│   │   ├── ChatInterface.tsx
│   │   └── Message.tsx
│   ├── lib/               # Utilities
│   │   └── api.ts         # API client
│   ├── store/             # State management
│   │   └── useAppStore.ts # Zustand store
│   ├── types/             # TypeScript types
│   │   └── index.ts
│   ├── App.tsx            # Main app component
│   ├── main.tsx           # Entry point
│   └── index.css          # Global styles
├── public/                # Static assets
├── index.html             # HTML template
├── vite.config.ts         # Vite configuration
├── tailwind.config.js     # Tailwind configuration
├── tsconfig.json          # TypeScript configuration
└── package.json           # Dependencies

```

## API Integration

The frontend communicates with the backend API at `http://localhost:8000`.

Vite proxy is configured to forward `/api/*` requests to the backend.

## Configuration

### Environment Variables

Create `.env` file (optional):

```env
VITE_API_URL=http://localhost:8000
```

### Theme Configuration

Tailwind config can be customized in `tailwind.config.js`.

## Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run type-check` - TypeScript type checking

## Technology Stack

### Core
- **React 18.2** - UI library
- **TypeScript 5.2** - Type safety
- **Vite 5.0** - Build tool

### UI & Styling
- **Tailwind CSS 3.3** - Utility-first CSS
- **lucide-react** - Icon library
- **clsx** - Conditional classnames

### State & Data
- **Zustand 4.4** - State management
- **@tanstack/react-query 5.12** - Data fetching
- **axios 1.6** - HTTP client

### Markdown & Code
- **react-markdown 9.0** - Markdown rendering
- **react-syntax-highlighter 15.5** - Syntax highlighting

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](../LICENSE)

---

**Built with ❤️ by Adrian Johnson**  
**Part of the Dual RAG LLM System**

