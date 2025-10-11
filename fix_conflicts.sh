#!/bin/bash
# Fix dependency conflicts in requirements.txt

cd /Users/wuy/Desktop/code/UAgent

echo "🔧 Fixing dependency conflicts in requirements.txt..."

# Fix urllib3 conflict (kubernetes requires <2.4.0)
sed -i '' 's/urllib3>=2.5.0/urllib3>=1.26.0,<2.4.0/' requirements.txt
echo "✅ Fixed urllib3 version constraint"

# Fix pyee conflict (playwright>=1.50.0 requires >=12)
sed -i '' 's/pyee==11.1.0/pyee>=12,<14/' requirements.txt
echo "✅ Fixed pyee version constraint"

# Fix greenlet conflict (playwright>=1.50.0 requires >=3.1.1)
sed -i '' 's/greenlet==3.0.3/greenlet>=3.1.1,<4.0.0/' requirements.txt
echo "✅ Fixed greenlet version constraint"

echo ""
echo "✅ All conflicts fixed!"
echo ""
echo "📋 Modified packages:"
echo "   • urllib3: >=1.26.0,<2.4.0 (compatible with kubernetes)"
echo "   • pyee: >=12,<14 (compatible with playwright>=1.50.0)"
echo "   • greenlet: >=3.1.1,<4.0.0 (compatible with playwright>=1.50.0)"
