import os
import warnings

import uvicorn


def main():
    # Suppress SyntaxWarnings from pydub.utils about invalid escape sequences
    warnings.filterwarnings('ignore', category=SyntaxWarning, module=r'pydub\.utils')

    # Support multiple environment variable names for port configuration
    # Priority: OPENHANDS_PORT > PORT > port (legacy) > default 3000
    port = int(
        os.environ.get('OPENHANDS_PORT') or
        os.environ.get('PORT') or
        os.environ.get('port') or
        '3000'
    )

    uvicorn.run(
        'openhands.server.listen:app',
        host='0.0.0.0',
        port=port,
        log_level='debug' if os.environ.get('DEBUG') else 'info',
    )


if __name__ == '__main__':
    main()
