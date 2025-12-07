# AI_IDS Frontend2 - React/TypeScript Frontend

Modern React-based frontend for the AI-Based Intrusion Detection System, built with TypeScript, Vite, and shadcn-ui.

## Features

- 🎨 **Modern UI**: Built with React, TypeScript, and Tailwind CSS
- 📁 **File Upload**: Drag & drop or click to upload PCAP/CSV files
- 📊 **Summary Statistics**: Visual breakdown of BENIGN, ATTACK, and ANOMALY flows
- 🔍 **Attack Type Breakdown**: Detailed view of detected attack types
- 💾 **CSV Export**: Download full prediction results
- 🔄 **Real-time API Integration**: Connected to the AI_IDS backend API

## Prerequisites

- Node.js 18+ and npm (or use [nvm](https://github.com/nvm-sh/nvm#installing-and-updating))
- Python 3.8+ (for the backend API)

## Quick Start

### 1. Install Dependencies

```bash
cd frontend2
npm install
```

### 2. Start the Backend API

The frontend requires the backend API to be running. From the project root:

```bash
python3 api/api_server.py
```

The API will run on `http://localhost:8000`

### 3. Start the Frontend

In a new terminal:

```bash
cd frontend2
npm run dev
```

The frontend will be available at `http://localhost:8080`

## Configuration

### API Base URL

By default, the frontend connects to `http://localhost:8000`. To change this:

1. Create a `.env` file in the `frontend2` directory:
```bash
VITE_API_BASE_URL=http://localhost:8000
```

2. Or modify `src/lib/api.ts` to change the default URL.

## Usage

1. **Upload a File**: 
   - Drag and drop a `.pcap`, `.pcapng`, or `.csv` file onto the upload area
   - Or click to browse and select a file

2. **Run Detection**: 
   - Click "Run Detection" to analyze the file
   - The system will process the file and show results

3. **View Results**:
   - Summary cards show counts for BENIGN, ATTACK, and ANOMALY flows
   - Attack types breakdown (if attacks are detected)
   - Processing information (file name, size, processing time)

4. **Download Results**:
   - Click "Download Full Results CSV" to save all predictions

## API Endpoints Used

- `GET /health` - Check API health status
- `POST /predict` - Upload file and get predictions
- `GET /download/{filename}` - Download CSV results

## Technologies

- **Vite** - Build tool and dev server
- **React 18** - UI framework
- **TypeScript** - Type safety
- **shadcn-ui** - UI component library
- **Tailwind CSS** - Styling
- **React Router** - Routing

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run build:dev` - Build in development mode
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Project Structure

```
frontend2/
├── src/
│   ├── components/      # React components
│   │   ├── ui/         # shadcn-ui components
│   │   ├── FileUpload.tsx
│   │   ├── SummaryResults.tsx
│   │   └── ...
│   ├── lib/            # Utilities
│   │   ├── api.ts      # API service
│   │   └── utils.ts
│   ├── pages/          # Page components
│   │   └── Index.tsx
│   └── main.tsx        # Entry point
├── public/             # Static assets
└── package.json
```

## Troubleshooting

### "API server is not responding"

- Ensure the backend API is running on port 8000
- Check that `VITE_API_BASE_URL` matches your backend URL
- Verify CORS is enabled on the backend (it should be by default)

### File upload fails

- Check file format (must be .pcap, .pcapng, or .csv)
- Ensure file size is reasonable
- Check browser console for errors
- Verify backend logs for processing errors

### Port 8080 already in use

- Change the port in `vite.config.ts`:
```typescript
server: {
  port: 8081, // or any available port
}
```

## Running Both Frontends

You can run both frontends simultaneously:

1. **Frontend (Original)**: `http://localhost:3000` - Vanilla JS frontend
2. **Frontend2 (React)**: `http://localhost:8080` - React/TypeScript frontend

Both connect to the same backend API at `http://localhost:8000`.

## Differences from Frontend (Original)

- **Technology**: React/TypeScript vs Vanilla JS
- **UI Framework**: shadcn-ui components vs custom CSS
- **Port**: 8080 vs 3000
- **Features**: Summary-focused view vs detailed table view
- **Build Tool**: Vite vs simple HTTP server

Both frontends are fully functional and can be used to demonstrate the system to different audiences.
