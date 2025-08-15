# 🌟 FLAI Website Documentation

Este directorio contiene el sitio web completo de FLAI con diseño cálido y profesional.

## 📁 Estructura del Sitio

```
docs/
├── index.html              # 🏠 Landing page principal
├── demo/index.html          # 📊 Demo interactiva
├── documentation.html       # 📖 Documentación completa
├── assets/styles.css        # 🎨 Estilos globales
└── README.md               # 📋 Esta documentación
```

## 🎯 Características Implementadas

### ✅ Landing Page (`index.html`)
- **Diseño cálido** con paleta terrosa/naranja en lugar de azul frío
- **Explicación clara** del problema y la solución
- **Secciones estructuradas**:
  - Hero con propuesta de valor
  - Problema vs Solución
  - Enfoque técnico (métrica 2D, grafo causal, mitigación)
  - Instalación y uso rápido
  - Investigación y papers
  - Recursos y comunidad
- **Navegación suave** con scroll automático
- **Responsive design** para móviles y tablets
- **Meta tags optimizados** para SEO

### ✅ Demo Interactiva (`demo/index.html`)
- **4 tabs organizadas**:
  1. 📊 Métrica Bidimensional - Explica EQA vs EQI
  2. 🔄 Flujo Completo - Workflow de mitigación
  3. 📚 Casos de Estudio - Adult, COMPAS, German Credit
  4. 💻 Código en Vivo - Simulación ejecutable
- **Ejemplos reales** alineados con el README
- **Datos y métricas** extraídos del repositorio
- **Código ejecutable** (simulado) en el navegador
- **Visualizaciones** de progreso y comparaciones

### ✅ Documentación (`documentation.html`)
- **API Reference completa** para ambas clases principales
- **Tabla de contenidos** sticky con navegación suave
- **Ejemplos de código** abundantes y prácticos
- **Mejores prácticas** y guías de optimización
- **Casos de uso completos** paso a paso
- **Interpretación de métricas** automatizada
- **Limitaciones y consideraciones** importantes

### ✅ Estilos Globales (`assets/styles.css`)
- **Tema cálido** con variables CSS organizadas
- **Paleta de colores** terrosa y profesional
- **Componentes reutilizables**: cards, botones, métricas
- **Animaciones suaves** y transiciones
- **Typography escalable** con clamp()
- **Grid layouts** responsivos
- **Estados de hover** y focus para accesibilidad

## 🔗 Navegación Integrada

Todas las páginas tienen navegación consistente:
- 🏠 **Inicio** → `index.html`
- 📊 **Demo** → `demo/index.html`  
- 📖 **Docs** → `documentation.html`
- 🐙 **GitHub** → repositorio externo
- 📦 **PyPI** → package externo

## 🚀 Cómo Usar

1. **Desarrollo local**: Abre cualquier HTML en un servidor local
2. **GitHub Pages**: Se sirve automáticamente desde `/docs`
3. **Actualización**: Edita los archivos HTML directamente

## 🎨 Personalización

### Cambiar Colores
Edita las variables CSS en `assets/styles.css`:
```css
:root {
  --accent: #ff7b54;        /* Color principal */
  --gold: #ffb347;          /* Color secundario */
  --bg: #1a1612;            /* Fondo principal */
  /* ... más variables */
}
```

### Añadir Contenido
- **Nuevas secciones**: Copia estructura de cards existentes
- **Nuevos ejemplos**: Añade en la demo o documentación
- **Nuevas métricas**: Actualiza tanto landing como demo

## 📊 Métricas Actualizadas

El sitio refleja los datos reales del repositorio:
- **EQI/EQA/F** del dataset Adult: -0.04, 0.08, 0.09
- **Comparaciones** antes/después de mitigación: 0.09 → 0.00
- **Casos reales**: Adult, COMPAS, German Credit
- **Código exacto** del README y notebooks

## 🔧 Mantenimiento

### Actualizar Enlaces
Los enlaces están centralizados en la navegación. Para cambios globales:
1. Edita el `nav` en cada página
2. Actualiza footer links
3. Verifica enlaces internos con `#anchors`

### Actualizar Contenido
1. **Nuevas versiones**: Cambiar números en todas las páginas
2. **Nuevos papers**: Añadir en sección investigación
3. **Nuevas features**: Documentar en API reference

### Testing
Verifica en diferentes navegadores:
- ✅ Chrome/Edge (Chromium)
- ✅ Firefox  
- ✅ Safari
- ✅ Móviles (responsive)

## 🎯 Mejoras Implementadas vs Versión Original

| Aspecto | Original | Nuevo |
|---------|----------|-------|
| **Diseño** | Frío, azul, minimalista | Cálido, terroso, acogedor |
| **Contenido** | Básico, técnico | Explicativo, problem-solution |
| **Demo** | Simple con PyScript | 4 tabs, ejemplos reales |
| **Documentación** | RST rota | HTML completa con ejemplos |
| **Navegación** | Desconectada | Integrada y cohesiva |
| **Mobile** | Limitado | Totalmente responsive |
| **SEO** | Básico | Meta tags optimizados |

## 📈 Próximos Pasos (Opcionales)

1. **Analytics**: Añadir Google Analytics para métricas
2. **Búsqueda**: Implementar búsqueda en documentación
3. **Blog**: Sección de artículos y tutoriales
4. **Feedback**: Sistema de comentarios o ratings
5. **i18n**: Versión en inglés

---

**✨ El sitio está listo para producción y representa fielmente el valor científico y técnico de FLAI.**